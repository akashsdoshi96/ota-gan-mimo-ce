mkdir data_16x64_A
pf_limit = 25;
no_of_iter = 240; %Number of channel realizations = pf_limit*no_of_iter
parpool(25);
for m = 1:no_of_iter
    SR = 15.36e6;
    T = SR*1e-3;
    nTxAnts = 64;
    nRxAnts = 16;
    NRB = 1;
    SCS = 15;
    nSlot = 0;
    hest = zeros(NRB*12,14,nRxAnts,nTxAnts,pf_limit);
    parfor n = 1:pf_limit
        seed = pf_limit*(m-1)+n;
        cdl = nrCDLChannel('Seed',seed);
        cdl.DelayProfile = 'CDL-A';
        cdl.DelaySpread = 30e-9;
        cdl.MaximumDopplerShift = 5;
        cdl.CarrierFrequency = 40e9;
        txsize = [nTxAnts,1,1,1,1];
        rxsize = [nRxAnts,1,1,1,1];
        cdl.TransmitAntennaArray.Size = txsize;
        cdl.ReceiveAntennaArray.ElementSpacing = [0.5 0.5 1 1];
        cdl.TransmitAntennaArray.ElementSpacing = [0.5 0.5 1 1];
        cdl.ReceiveAntennaArray.Size = rxsize;
        cdl.SampleRate = SR;
        cdlInfo = info(cdl);
        Nt = cdlInfo.NumTransmitAntennas;
        in = complex(randn(T,Nt),randn(T,Nt));
        [~,pathGains,sampleTimes] = cdl(in);
        pathFilters = getPathFilters(cdl);
        offset = nrPerfectTimingEstimate(pathGains,pathFilters);
        hest(:,:,:,:,n) = nrPerfectChannelEstimate(pathGains,pathFilters,...
            NRB,SCS,nSlot,offset,sampleTimes);
    end
    for n = 1:pf_limit
        a = hest(1,1,:,:,n);
        save_var = pf_limit*(m-1)+n;
        fname = sprintf('data_16x64_A/hA%d.mat',save_var);
        save(fname,'a');
    end
    clear hest;
end