% CLASS: QuantizeEstimOut
%
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
%
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The QuantizeEstimOut class defines a scalar observation channel, p(y|z),
%   that constitutes a scalar quantization model, i.e., y is an
%   element of the set {0,1,...2^b-1}, z is a real number, and
%             	 p(y = i | z) = Phi[(z - t_i - Mean)/sqrt(Var)] - Phi[(z - t_(i+1))/sqrt(Var)],
%   where t_i, 0 <= i <= 2^b is the quantization threshold (t_0 = -infty, t_(2^b) = +infty) and
%   Phi((x-b)/sqrt(c)) is the cumulative density function (CDF) of a
%   Gaussian random variable with mean b, variance c, and argument x.
%   Typically, mean = 0 and var = 1, thus
%   p(y = i | z) = Phi[(z - t_i)/sqrt(Var)] - Phi[(z - t_(i+1))/sqrt(Var)].
%
% PROPERTIES (State variables)
%   Y                   An M-by-T array of scalar quantization output (0,1,..., 2^b-1) for the
%                       training data, where M is the number of training data
%                       points, and T is the number of classifiers being learned
%                       (typically T = 1)
%   Quantize_bit        a scalar denoting the number of quantization
%                       bits
%   Quantize_stepsize  a scalar denoting the quantization stepsize
%   Mean                [Optional] An M-by-T array of noise means (see
%                       DESCRIPTION) [Default: 0]
%   Var                 [Optional] An M-by-T array of noise variances
%                       (see DESCRIPTION).  Note that smaller variances equate to
%                       more step-like sigmoid functions [Default: 1e-2]
%   maxSumVal           Perform MMSE estimation (false) or MAP estimation (true)?
%                       [Default: false] Currently, only support the MMSE
%                       estimation!!!
%
% METHODS (Subroutines/functions)
%   QuantizeEstimOut(Y, Quantize_bit, Quantize_stepsize)
%       - Default constructor.  Assigns Mean and Var to default values.
%   QuantizeEstimOut(Y, Quantize_bit, Quantize_stepsize, Mean)
%       - Optional constructor.  Sets both Y and Mean.
%   QuantizeEstimOut(Y, Quantize_bit, Quantize_stepsize, Mean, Var)
%       - Optional constructor.  Sets Y, Mean, and Var.
%   QuantizeEstimOut(Y, Quantize_bit, Quantize_stepsize, Mean, Var, maxSumVal)
%       - Optional constructor.  Sets Y, Mean, Var, and maxSumVal.
%   estim(obj, zhat, zvar)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the quantization model and maxSumVal = false (see
%         DESCRIPTION), and p(z) = Normal(zhat,zvar).
%

%
% Coded by: Jianhua Mo, The University of Texas at Austin.
% E-mail: mojianhua01@gmail.com
% Last change: 06/02/2015
% Change summary:
%       - Created (06/02/15; Jianhua Mo)
%       - Modified estim method when Prob(y=i |z) is small
%         (v0.1) (06/08/15; Jianhua Mo)
%       - Modified by Prof. Schniter to speed up the codes (v0.2) (12/20/16)
% Version 0.2
%

classdef QuantizeEstimOut < EstimOut
    
    properties
        Y;                          % M-by-T vector of quantization output labels
        Quantize_bit;               % # of quantization bits
        Quantize_stepsize;          % thresholds of the quantization
        Mean = 0;       % M-by-T vector of noise means [dflt: 0]
        Var = 1e-2;    	% M-by-T vector of noise variances [dflt: 1e-2]
        maxSumVal = false;   % Sum-product (false) or max-sum (true) GAMP? Currently only support sum-product GAMP
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = QuantizeEstimOut(Y, Quantize_bit, Quantize_stepsize, Mean, Var, maxsumval)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.Quantize_bit = Quantize_bit;            % set Quantize_bit
                obj.Quantize_stepsize = Quantize_stepsize;  % set Quantize_stepsize
                obj.Y = Y;      % Set Y
                if nargin >= 4 && ~isempty(Mean)
                    % Mean property is an argument
                    obj.Mean = Mean;
                end
                if nargin >= 5 && ~isempty(Var)
                    % Var property is an argument
                    obj.Var = Var;
                end
                if nargin >= 6 && ~isempty(maxsumval)
                    % maxSumVal property is an argument
                    obj.maxSumVal = maxsumval;
                end
                if any(Var(:) < 0) && ~obj.maxSumVal
                    error('Var must be non-negative when running sum-product GAMP')
                elseif any(Var(:) <= 0) && obj.maxSumVal
                    error('Var must be strictly positive when running max-sum GAMP')
                end
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.Y(obj, Y)
            % Elements of Y must be either in {0,1,...2^b-1}
            obj.Y = Y;
        end
        
        function obj = set.Quantize_bit(obj, Quantize_bit)
            obj.Quantize_bit = Quantize_bit;
        end
        
        function obj = set.Quantize_stepsize(obj, Quantize_stepsize)
            obj.Quantize_stepsize = Quantize_stepsize;
        end
        
        function obj = set.Mean(obj, Mean)
            obj.Mean = double(Mean);
        end
        
        function obj = set.Var(obj, Var)
            if any(Var(:) < 0)
                error('Var must be non-negative')
            else
                obj.Var = double(Var);
            end
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            if isscalar(maxsumval) && islogical(maxsumval)
                obj.maxSumVal = maxsumval;
            else
                error('QuantizeEstimOut: maxSumVal must be a logical scalar')
            end
        end
        
        
        % *****************************************************************
        %                          ESTIM METHOD
        % *****************************************************************
        % This function will compute the posterior mean and variance of a
        % random vector Z whose prior distribution is N(Phat, Pvar), given
        % observations Y obtained through the separable channel model:
        % p(Y(m,t) = i | Z(m,t)) = Phi[ (Z(m,t) - t_i -
        % Mean(m,t)) / sqrt(Var(m,t)) ] - Phi[ (Z(m,t) - t_(i+1) -
        % Mean(m,t)) / sqrt(Var(m,t)) ]
        % if obj.maxSumVal = false.
        function [Zhat, Zvar] = estim(obj, Phat, Pvar)
            %             global z;
            switch obj.maxSumVal
                case false
%                     if obj.Quantize_bit == 1
%                         % Return sum-product expressions to GAMP in Zhat and
%                         % Zvar
%                         
%                         t_lower =  (- 2^( obj.Quantize_bit -1 ) + obj.Y )* obj.Quantize_stepsize;       % lower threshold
%                         t_upper =  (- 2^( obj.Quantize_bit -1 ) + obj.Y + 1)* obj.Quantize_stepsize;    % uppper threshold
%                         
%                         t_lower(obj.Y == 0) = -inf;
%                         t_upper(obj.Y == 2^(obj.Quantize_bit)-1 ) = +inf;
%                         
%                         z_lower = (( Phat - obj.Mean - t_lower ) ./sqrt(Pvar + obj.Var));
%                         z_upper = (( Phat - obj.Mean - t_upper ) ./sqrt(Pvar + obj.Var));
%                         
%                         % Now compute the probability P(Y == i)
%                         % Prob = normcdf(z_lower) - normcdf(z_upper);
%                         % Prob = 1/2 * erfc( - z_lower / sqrt(2) ) - 1/2 * erfc( - z_upper / sqrt(2) );
%                         
%                         I1 = find(t_lower == -inf);
%                         I2 = find(t_upper == +inf);
%                         
%                         Ratio_upper = zeros( size(obj.Y) );
%                         Ratio_lower = zeros( size(obj.Y) );
%                         
%                         Ratio_upper(I1) = ( (2/sqrt(2*pi)) * (erfcx( z_upper(I1) / sqrt(2)).^(-1)) );
%                         Ratio_lower(I2) = ( (2/sqrt(2*pi)) * (erfcx(-z_lower(I2) / sqrt(2)).^(-1)) );
%                         
%                         Zhat(I1) = Phat(I1) - ( (Pvar(I1) ./ sqrt(Pvar(I1) + obj.Var)).* Ratio_upper(I1) );
%                         
%                         Zhat(I2) = Phat(I2) + ( (Pvar(I2) ./ sqrt(Pvar(I2) + obj.Var)).* Ratio_lower(I2) );
%                         
%                         Zvar(I1) = Pvar(I1) - (Pvar(I1).^2 ./ (Pvar(I1) + obj.Var)) .*...
%                             Ratio_upper(I1).*(-z_upper(I1) + Ratio_upper(I1));
%                         Zvar(I2) = Pvar(I2) - (Pvar(I2).^2 ./ (Pvar(I2) + obj.Var)) .*...
%                             Ratio_lower(I2).*( z_lower(I2) + Ratio_lower(I2));
%                         
%                         Zhat = Zhat';
%                         Zvar = Zvar';
                        
%                     else
                        % multi-bit ADCs
                        % Return sum-product expressions to GAMP in Zhat and Zvar
                        
                        t_lower =  (- 2^( obj.Quantize_bit -1 ) + obj.Y )* obj.Quantize_stepsize;       % lower threshold
                        t_upper =  (- 2^( obj.Quantize_bit -1 ) + obj.Y + 1)* obj.Quantize_stepsize;    % upper threshold
                        
                        t_lower(obj.Y == 0) = -inf;
                        t_upper(obj.Y == 2^(obj.Quantize_bit)-1 ) = +inf;

                        inv_sqrt_Pvar_plus_Var = 1./sqrt(Pvar + obj.Var);
                        z_lower = ( Phat - obj.Mean - t_lower ) .* inv_sqrt_Pvar_plus_Var;
                        z_upper = ( Phat - obj.Mean - t_upper ) .* inv_sqrt_Pvar_plus_Var;
                        
                        % Now compute the probability P(Y == i)
                        % Prob = normcdf(z_lower) - normcdf(z_upper);
                        Prob = 0.5 * erfc( - (1/sqrt(2))*z_lower ) - 0.5 * erfc( - (1/sqrt(2))*z_upper );
                        
                        Ratio_lower = normpdf(z_lower)./Prob;
                        Ratio_upper = normpdf(z_upper)./Prob;
                        
                        % Finally, compute E[Z(m,t) | Y(m,t)] = Zhat, and
                        % var{Z(m,t) | Y(m,t)} = Zvar
                        Zhat = Phat +  (Pvar .* inv_sqrt_Pvar_plus_Var) .* (Ratio_lower - Ratio_upper );
                        
                        Pvar2_over_Pvar_plus_Var = Pvar.^2 ./ (Pvar + obj.Var); 
                        Zvar = Pvar - Pvar2_over_Pvar_plus_Var .* Ratio_lower.*(z_lower + Ratio_lower)...
                                    - Pvar2_over_Pvar_plus_Var .* Ratio_upper.*(-z_upper + Ratio_upper)...
                                 + 2* Pvar2_over_Pvar_plus_Var .* Ratio_lower.* Ratio_upper;
                        
                        
                        I = find(Prob <= 1e-8); % deal with the case with extremely small Prob
                        if ~isempty(I)
                            Zhat(I) = Phat(I) + (Pvar(I) ./ (Pvar(I) + obj.Var + 1/12 * (obj.Quantize_stepsize)^2 )).*...
                                ( obj.Mean + 0.5*(t_lower(I) + t_upper(I)) - Phat(I));  %% I made a mistake in this line before!!!
                            
                            Zvar(I) = Pvar(I).^2 ./ (Pvar(I) + obj.Var + 1/12 * (obj.Quantize_stepsize)^2 );
                        end;
                        
                        if any(Zvar(:)<=0 )
                            keyboard;
                        end;
                        
                        I1 = find(t_lower == -inf);
                        I2 = find(t_upper == +inf);
                        
                        Ratio_upper(I1) = (2/sqrt(2*pi)) * (erfcx( z_upper(I1) / sqrt(2)).^(-1));
                        Ratio_lower(I2) = (2/sqrt(2*pi)) * (erfcx(-z_lower(I2) / sqrt(2)).^(-1));
                        
                        Zhat(I1) = Phat(I1) - ( (Pvar(I1) ./ sqrt(Pvar(I1) + obj.Var)).* Ratio_upper(I1) );
                        
                        Zhat(I2) = Phat(I2) + ( (Pvar(I2) ./ sqrt(Pvar(I2) + obj.Var)).* Ratio_lower(I2) );
                        
                        Zvar(I1) = Pvar(I1) - (Pvar(I1).^2 ./ (Pvar(I1) + obj.Var)) .*...
                            Ratio_upper(I1).*(-z_upper(I1) + Ratio_upper(I1));
                        Zvar(I2) = Pvar(I2) - (Pvar(I2).^2 ./ (Pvar(I2) + obj.Var)) .*...
                            Ratio_lower(I2).*(z_lower(I2) + Ratio_lower(I2));
                        
                        
                        % For the cases other than 'random_QPSK'
                        %                     Zhat = max(min(Zhat, 4 * 2^( obj.Quantize_bit -1)* obj.Quantize_stepsize), ...
                        %                         - 4 * 2^( obj.Quantize_bit -1)* obj.Quantize_stepsize);
                        
                        if any(Zvar(:)<=0 | Zvar(:)== +inf | isnan(Zvar(:)))
                            keyboard;
                        end;
%                     end;
                    
                case true
                    % Return max-sum expressions to GAMP in Zhat and Zvar
                    error('This estimator is not implemented for maxsum')
            end
        end
        
        
        % *****************************************************************
        %                         LOGLIKE METHOD
        % *****************************************************************
        % This function will compute *an approximation* to the expected
        % log-likelihood, E_z[log p(y|z)] when performing sum-product GAMP
        % (obj.maxSumVal = false).  The approximation is based on Jensen's
        % inequality, i.e., computing log E_z[p(y|z)] instead.  If
        % performing max-sum GAMP (obj.maxSumVal = true), logLike returns
        % log p(y|z) evaluated at z = Zhat
        function ll = logLike(obj, Zhat, Zvar)
            
            switch obj.maxSumVal
                case false
                    
                    t_lower =  (- 2^( obj.Quantize_bit -1 ) + obj.Y )* obj.Quantize_stepsize;       % upper threshold
                    t_upper =  (- 2^( obj.Quantize_bit -1 ) + obj.Y + 1)* obj.Quantize_stepsize;    % lower threshold
                    
                    t_lower( obj.Y == 0) = -inf;
                    t_upper( obj.Y == 2^(obj.Quantize_bit)-1 ) = +inf;
                    
                    inv_sqrt_Pvar_plus_Var = 1./sqrt(Pvar + obj.Var);
                        z_lower = ( Phat - obj.Mean - t_lower ) .* inv_sqrt_Pvar_plus_Var;
                        z_upper = ( Phat - obj.Mean - t_upper ) .* inv_sqrt_Pvar_plus_Var;
                    
%                     C_lower = (( Zhat - obj.Mean - t_lower ) ./sqrt(Zvar + obj.Var));
%                     C_upper = (( Zhat - obj.Mean - t_upper ) ./sqrt(Zvar + obj.Var));
                    
                    % Now compute the probability P(Y == i)
                    % Prob = normcdf(C_lower) - normcdf(C_upper);
                    Prob = 0.5 * erfc( - C_lower / sqrt(2) ) - 0.5 * erfc( - C_upper / sqrt(2) );
                    
                    ll = log(Prob);
                    
                    I = find(Prob < 1e-4);
                    
                    %                     Prob(I) = 1/2 * exp( -C_lower(I).^2/2 ) .* erfcx( - C_lower(I)/sqrt(2) ) - ...
                    %                         1/2 * exp( -C_upper(I).^2/2 ) .* erfcx( - C_upper(I)/sqrt(2) );
                    %
                    %                     ll(I) = log(Prob(I));
                    
                    ll(I) = -log( sqrt(2*pi)) - ( (C_lower(I) + C_upper(I))/2 ).^2/2 + ...
                        log(C_lower(I) - C_upper(I));
                    
                    I1 = find(t_lower == -inf);
                    I2 = find(t_upper == +inf);
                    
                    C(I1,1) = - C_upper(I1);
                    C(I2,1) = C_lower(I2);
                    
                    % ll(I1) = log(normcdf(C(I1), 0, 1));
                    % ll(I2) = log(normcdf(C(I2), 0, 1));
                    
                    ll(I1) = log( 1/2* erfc ( - C(I1)/ sqrt(2)));
                    ll(I2) = log( 1/2* erfc ( - C(I2)/ sqrt(2)));
                    
                    I = find(C < -30);
                    %This expression is equivalent to log(normpdf(C)) for
                    %all negative arguments greater than -38 and is
                    %numerically robust for all values smaller.  DO NOT USE
                    %FOR LARGE positive x.
                    ll(I) = -log(2)-0.5*C(I).^2+log(erfcx(-C(I)/sqrt(2)));
                    
                    %                     ll(I1) = -log(2)-0.5*C_upper(I1).^2+log(erfcx( C_upper(I1)/sqrt(2)));
                    %                     ll(I2) = -log(2)-0.5*C_lower(I2).^2+log(erfcx(-C_lower(I2)/sqrt(2)));
                    
                    %                     if any( ll(:) == +inf | ll(:)== -inf | isnan( ll(:)) )
                    %                         keyboard;
                    %                     end;
                case true
                    error('This estimator is not implemented for maxsum')
            end
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)   % not done!!!
            keyboard;
            error('logScale method not implemented for this class. Set the GAMP option adaptStepBethe = false.');
            
            %Find sign needed to compute the inner factor
            PMonesY = sign(obj.Y - 0.1);	% +/- 1 for Y(m,t)'s
            
            if~(obj.maxSumVal)
                % Find the fixed-point of phat
                opt.phat0 = Axhat;
                opt.alg = 1; % approximate newton's method
                opt.maxIter = 40;
                opt.tol = 1e-4;
                opt.threshold = 0.2;
                %Smallest regularization setting without getting the warnings for Var= 0
                opt.regularization = 1e-6;
                opt.debug = false;
                phatfix = estimInvert(obj,Axhat,pvar,opt);
                
                C = PMonesY .* ((phatfix - obj.Mean) ./ sqrt(pvar + obj.Var));
                % Compute log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
                % Has a closed form solution: see C. E. Rasmussen,
                % "Gaussian Processes for Machine Learning." sec 3.9.
                % ls = log(normcdf(C, 0 , 1));
                ls = log( 1/2 * erfc(-C / sqrt(2)) );
                
                %Find bad values that cause log cdf to go to infinity
                I = find(C < -30);
                %This expression is equivalent to log(normpdf(C)) for
                %all negative arguments greater than -38 and is
                %numerically robust for all values smaller.  DO NOT USE
                %FOR LARGE positive x.
                ls(I) = -log(2)-0.5*C(I).^2+log(erfcx(-C(I)/sqrt(2)));
                
                % Combine to form output cost
                ll = ls + 0.5*(Axhat - phatfix).^2./pvar;
            else
                %Compute true constant
                C = PMonesY .* (Axhat - obj.Mean)/sqrt(obj.Var);
                %ll = log(normcdf(C, 0, 1));
                ll = log( 1/2 * erfc(-C /sqrt(2)) );
                
                %Find bad values that cause log cdf to go to infinity
                I = find(C < -30);
                %This expression is equivalent to log(normpdf(C)) for
                %all negative arguments greater than -38 and is
                %numerically robust for all values smaller.  DO NOT USE
                %FOR LARGE positive x.
                ll(I) = -log(2)-0.5*C(I).^2+log(erfcx(-C(I)/sqrt(2)));
            end
            
        end
        
        % Return number of columns of Y
        function S = numColumns(obj)
            S = size(obj.Y, 2);
        end

        % Return size of Y
        function [nz,ncol] = size(obj)
            [nz,ncol] = size(obj.Y);
        end
        
        % Generate random samples given the distribution
        function y = genRand(obj, z)   % not done!!!
            if obj.Var > 0
                actProb = normcdf(z,obj.Mean, sqrt(obj.Var));
                y = (rand(size(z)) < actProb);
            elseif obj.Var == 0
                y = sign(z-obj.Mean);
                y(y == -1) = 0;
            else
                error('Var must be non-negative')
            end
        end
    end
end
