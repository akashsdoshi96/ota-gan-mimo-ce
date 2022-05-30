classdef TuneVarEstimIn < EstimIn
    % TuneVarEstimIn:  Include variance-tuning in input estimator 

    properties
        est;             % base estimator
        tuneDim = 'col'; % dimension on which the variances are to match
        nit = 4;         % number of EM iterations 
        rvarHist;        % history of rvar variances
    end
    
    methods
        % Constructor
        function obj = TuneVarEstimIn(est,varargin)
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.est = est;
                for i=1:2:length(varargin) 
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
        
        % Compute prior mean and variance
        function [xhat, xvar, valInit] = estimInit(obj)
            [xhat, xvar, valInit] = obj.est.estimInit;
        end
        
        % Compute posterior mean and variance from Gaussian estimate
        function [xhat, xvar] = estim(obj, rhat, rvar)
            rvar1 = rvar;
            obj.rvarHist = zeros(obj.nit,2);
            for it = 1:obj.nit
                
                [xhat, xvar1] = obj.est.estim(rhat,rvar1);
                
                [N,T] = size(rhat);
                rvar2 = abs(xhat-rhat).^2 + xvar1;
                obj.rvarHist(it,1) = mean(rvar1(:));
                obj.rvarHist(it,2) = mean(rvar2(:));
                                
                switch obj.tuneDim
                    case 'joint'
                        rvar1 = sum(rvar2(:))/N/T;
                    case 'col'
                        rvar1 = repmat(sum(rvar2)/N,[N 1]);
                    case 'row'
                        rvar1 = repmat(sum(rvar2,2)/T, [1 T]);
                    otherwise
                        error('Invalid tuning dimension in TuneVarEstimIn');
                end                
            end
            xvar = xvar1.*rvar./rvar1;
        end
        
        % Generate random samples
        function x = genRand(obj, nx)
            x = obj.est.genRand(nx);
        end
        
    end
    
end

