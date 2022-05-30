% CLASS: CQuantizeEstimOut
%
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
%
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The CQuantizeEstimOut class defines a scalar observation channel, p(y|z),
%   that constitutes a probit binary classification model, i.e., y is an
%   element of the set {0,1}, z is a real number, and
%             	 p(y = 1 | z) = Phi((z - Mean)/sqrt(Var)),
%   where Phi((x-b)/sqrt(c)) is the cumulative density function (CDF) of a
%   Gaussian random variable with mean b, variance c, and argument x.
%   Typically, mean = 0 and var = 1, thus p(y = 1 | z) = Phi(z).
%
% PROPERTIES (State variables)
%   Y           An M-by-T array of binary ({0,1}) class labels for the
%               training data, where M is the number of training data
%               points, and T is the number of classifiers being learned
%               (typically T = 1)
%   Mean        [Optional] An M-by-T array of probit function means (see
%               DESCRIPTION) [Default: 0]
%   Var         [Optional] An M-by-T array of probit function variances
%               (see DESCRIPTION).  Note that smaller variances equate to
%               more step-like sigmoid functions [Default: 1e-2]
%   maxSumVal 	Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: false]
%
% METHODS (Subroutines/functions)
%   CQuantizeEstimOut(Y)
%       - Default constructor.  Assigns Mean and Var to default values.
%   CQuantizeEstimOut(Y, Mean)
%       - Optional constructor.  Sets both Y and Mean.
%   CQuantizeEstimOut(Y, Mean, Var)
%       - Optional constructor.  Sets Y, Mean, and Var.
%   CQuantizeEstimOut(Y, Mean, Var, maxSumVal)
%       - Optional constructor.  Sets Y, Mean, Var, and maxSumVal.
%   estim(obj, zhat, zvar)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the probit model and maxSumVal = false (see
%         DESCRIPTION), and p(z) = Normal(zhat,zvar).  When maxSumVal =
%         true, estim returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z).
%

%
% Coded by: Jianhua Mo, The University of Texas at Austin.
% E-mail: mojianhua01@gmail.com
% Last change: 06/02/2015
% Change summary:
%       - Created (06/02/15; Jianhua Mo)
% Version 0.0
%

classdef CQuantizeEstimOut < EstimOut
    
    properties
        Y;              % M-by-T vector of binary class labels
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
        function obj = CQuantizeEstimOut(Y, Quantize_bit, Quantize_stepsize, Mean, Var, maxsumval)
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
            Y_real = real(Y);
            Y_imag = imag(Y);
            % Elements of Y must be either in {0,1,...2^b-1}
            obj.Y = Y_real + 1j* Y_imag;
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
                error('CQuantizeEstimOut: maxSumVal must be a logical scalar')
            end
        end
        
        
        % *****************************************************************
        %                          ESTIM METHOD
        % *****************************************************************
        function [Zhat, Zvar] = estim(obj, Phat, Pvar)
            if isreal(Pvar) == false
                keyboard;
            end;
            
            % assume the real and imaginary parts have the same variance
            obj_real = QuantizeEstimOut(real(obj.Y), obj.Quantize_bit, obj.Quantize_stepsize, real(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            obj_imag = QuantizeEstimOut(imag(obj.Y), obj.Quantize_bit, obj.Quantize_stepsize, imag(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            [Zhat_real, Zvar_real] = estim(obj_real, real(Phat), 1/2*Pvar);
            [Zhat_imag, Zvar_imag] = estim(obj_imag, imag(Phat), 1/2*Pvar);
            Zhat = Zhat_real + 1j* Zhat_imag;
            Zvar = Zvar_real + Zvar_imag;
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
            %             keyboard
            obj_real = QuantizeEstimOut(real(obj.Y), obj.Quantize_bit, obj.Quantize_stepsize, real(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            obj_imag = QuantizeEstimOut(imag(obj.Y), obj.Quantize_bit, obj.Quantize_stepsize, imag(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            ll_real = logLike(obj_real, real(Zhat), 1/2*Zvar);
            ll_imag = logLike(obj_imag, imag(Zhat), 1/2*Zvar);
            ll = ll_real + ll_imag;  % sum of log-likelihood of the real and imaginary parts
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
            
            obj_real = QuantizeEstimOut(real(obj.Y), real(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            obj_imag = QuantizeEstimOut(imag(obj.Y), imag(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            ll_real = logScale(obj_real, real(Axhat), 1/2*pvar, real(phat));
            ll_imag = logScale(obj_imag, imag(Axhat), 1/2*pvar, imag(phat));
            ll = ll_real + ll_imag;      % sum of log-likelihood of the real and imaginary parts
        end
        
        
        % *****************************************************************
        %                       NUMCOLUMNS METHOD
        % *****************************************************************
        function S = numColumns(obj)
            % Return number of columns of Y
            S = size(obj.Y, 2);
        end
        
        
        % *****************************************************************
        %                       SIZE METHOD
        % *****************************************************************
        function [M,N] = size(obj)
            % Return size of Y
            [M,N] = size(obj.Y);
        end
        
        %Generate random samples given the distribution
        function y = genRand(obj, z)
            obj_real = CQuantizeEstimOut(real(obj.Y), real(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            obj_imag = CQuantizeEstimOut(imag(obj.Y), imag(obj.Mean), 1/2*obj.Var, obj.maxSumVal);
            y_real = genRand(obj_real, real(z));
            y_imag = genRand(obj_imag, imag(z));
            y = y_real  + 1j* y_imag;
        end
    end
end