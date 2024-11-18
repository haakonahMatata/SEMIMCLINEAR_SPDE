# Code used to generate figures 1,2 and 3 in manuscript.
import Pkg
Pkg.add("Distributions");

#Pkg.add("FFTW");
Pkg.add("JLD2");
# #Pkg.add("PLots")
# Pkg.add("PlotlyBase")
# Pkg.add("PlotlyJS")
Pkg.add("CairoMakie")
#Pkg.add("LaTeXStrings")
Pkg.add("MakieTeX")
using JLD2
using Distributed
#using Plots
#pythonplot()
using CairoMakie
#using LaTeXStrings
using MakieTeX

parallel_procs=50;#set to number of processors on machine (if memory is an issue, you may need to use less)
if nprocs()<parallel_procs
    addprocs(parallel_procs-nprocs()+1);
end
const workers = nprocs()-1;
println("workers $workers")
@everywhere using Distributions
@everywhere using FFTW;
@everywhere using Random
import Future.randjump


#rcParams["text.latex.preamble"] = L"\usepackage{siunitx} \usepackage{amsfonts}"

#= PROBLEM SETTING
We consider the problem of solving the stochastic differential equation 
du =  ( ϵ* Delta u + u)  dt + sigma B dw  (t,x) in [0,tau] x (0,1)
u(t=0, x) =u_0(x)
u_x(t,0) = u_x(t,1) =0   t in [0,tau]

and solve spectrally using a sine series basis
=#


#κ=1 and β2=ν*κ = ν for all trial cases
@everywhere function linearproblem(;ν=1)

    f(u) = @. u
    g(u) = @. 0;
    σ =1.
    ϵ = 0.2
    τ = 1
    
    #ν = 1;
    β1 = 2;
    

    
    #append maginutde of b to pName (pName is used for storing results to file)
    pName = "linAddnoiseNu"*string(round(ν,sigdigits=2))*"eps"*string(ϵ);
    print(pName)
    return β1, ϵ, ν, σ, τ, f, g, pName
end

@everywhere function linear_multNoiseProblem(;ν=1)

    f(u) = @. u
    g(u) = @. u;
    σ =1.
    ϵ = 0.2
    τ = 1
    β1 = 1;
    
    #append maginutde of b to pName (pName is used for storing results to file)
    pName = "multNoiseNu"*string(round(ν,sigdigits=2))*"eps"*string(ϵ);
    print(pName)
    return β1, ϵ, ν, σ, τ, f, g, pName;
end

@everywhere function linear_multNoiseProblemShifted(;ν=1)

    f(u) = @. u
    g(u) = @. [u[2:end];0];
    σ =1.
    ϵ = 0.2
    τ = 1
    β1 = 1;
    
    #append maginutde of b to pName (pName is used for storing results to file)
    pName = "multNoiseNuShifted"*string(round(ν,sigdigits=2))*"eps"*string(ϵ);
    print(pName)
    return β1, ϵ, ν, σ, τ, f, g, pName;
end



function main()
    ## In the main just call the various
    #runalltests(linearproblem(ν=2))
    #runalltests(linearproblem(ν=4//3))
    #runalltests(linearproblem(ν=1))

    #runalltests(linear_multNoiseProblem(ν=2))
    #runalltests(linear_multNoiseProblem(ν=4//3))
    #runalltests(linear_multNoiseProblem(ν=1))

    
    #runalltests(linear_multNoiseProblemShifted(ν=2))
    runalltests(linear_multNoiseProblemShifted(ν=4//3))
    #runalltests(linear_multNoiseProblemShifted(ν=1))
        
end

function runalltests(pData)
    ## run all tests and store output results
    
    # setup
    rng = MersenneTwister(123457)
       
    # creating rng seeds for parallel computations
    rngVec = Array{MersenneTwister}(undef, workers);
    rngVec[1] = rng;
    for i=2:workers
        rngVec[i] = randjump(rngVec[i-1], big(10)^20);
    end
    M  = 1000

    #doubleDifferenceRates(pData,rngVec,M);
    #mlmcDifferenceRates(pData,rngVec,M);
    mlmctest(pData, rngVec)
        
end

@everywhere function doubleDifference(UFF,UFC,UCF,UCC)
    NF = length(UFF);
    NC = length(UCC);
    return (UFF-UFC-vcat(UCF-UCC, zeros(NF-NC)));
end

@everywhere function singleDifference(UFF,UCC)
    NF = length(UFF);
    NC = length(UCC);
    return (UFF-vcat(UCC, zeros(NF-NC)));
end

function skiprngsahead(rngVec)
    for i=1:workers
        rngVec[i] = randjump(rngVec[i], big(10)^15);
    end
end

@everywhere function coupledSolvers(method, NF, NC, JF, JC,pData, rng,M)
    # coupled solvers linear problem
    # method ==1 is mimc, ==2 is mlmc
    β1,ϵ, ν, σ, τ, f,g, pName = pData

    # This solver encompasses all solvers for the LINEAR REACTION TERM SETTING.
    
    ΔtF = τ/JF; ΔtC = τ/JC;
    #if initial data is random, initialization must be coupled, but for problems with 
    # deterministic initial data we can write    

    ## FF - Fine space Fine time, FC - fine space, coarse time,
    ## CF - coarse space, fine time, CC - coarse space coarse time 
        
    kF                 = (1:NF);
    λF                 =  ϵ*kF.^ν;#ϵ*kF.^2;
    μF                 =  1. ./kF.^(1.01);
    αF                 = μF.^(-1/2.)
    driftLapFF         = exp.(-λF*ΔtF);
    driftReactionFF    = (1 .- exp.(-λF*ΔtF)) ./λF;
    stdNoiseF          = σ*sqrt.(μF.*(1 .-exp.(-2*λF*ΔtF))./(2*λF));    
       
    driftLapFC       = exp.(-λF*ΔtC);
    driftReactionFC  = (1 .- exp.(-λF*ΔtC)) ./λF;

    #kC             = kF[1:NC]
    λC              = λF[1:NC];
    αC              = αF[1:NC];
    driftLapCF      = exp.(-λC*ΔtF);
    driftReactionCF = (1 .- exp.(-λC*ΔtF)) ./λC;
    driftLapCC      = exp.(-λC*ΔtC);
    driftReactionCC = (1 .- exp.(-λC*ΔtC)) ./λC;

    ξF1 = zeros(NF);
    ξF2 = zeros(NF);    
    ξC = zeros(NC);    
    ξC1 = zeros(NC);
    ξC2 = zeros(NC);    

    ΔUSum           = zeros(NF);
    errFinalTimeSum = 0;
    #errAllTimeSum   = zeros(JC);

    
    for i=1:M

        # Solutions: 
        # UFF - Fine space, Fine time
        # UFC - Fine space, Coarse time
        # UCF - Coarse space, Fine time
        # UCC - Coarse space, Coarse time
        
        UFF = zeros(NF); #randn(rng,NF) ./(kF).^2.;#(kF).^1.5;
        #UFF = U₀(NF,rng);
        
        if NC < NF 
            UCF = deepcopy(UFF[1:NC]);
        end

        if JC < JF 
            UFC = deepcopy(UFF);
        end

        if NC < NF && JC<JF
            UCC = deepcopy(UFF[1:NC]);
        end


        
        for j=1:(JF÷2)
                        
            ξF1 = stdNoiseF.*randn(rng, NF);
            ξF2 = stdNoiseF.*randn(rng, NF); 
            ξC1 = ξF1[1:NC];
            ξC2 = ξF2[1:NC];

            # One step fine solutions time
            UFF             = driftLapFF.*UFF + driftReactionFF.*f(UFF) + (1 .+αF.*g(UFF)).*ξF1 ;
            if NC < NF
                UCF         = driftLapCF.*UCF + driftReactionCF.*f(UCF) + (1 .+αC.*g(UCF)).*ξC1;
            end
            #ΔU              = doubleDifference(UFF,UFC,UCF,UCC);
            #errAllTimeSum[2*j-1]  += sum(ΔU.^2); 
            
            # Another step, fine solutions in time                        
            UFF                = driftLapFF.*UFF + driftReactionFF.*f(UFF) + (1 .+αF.*g(UFF)).*ξF2;
            if NC < NF
                UCF            = driftLapCF.*UCF + driftReactionCF.*f(UCF) + (1 .+αC.*g(UCF)).*ξC2;
            end

            
            if JC < JF
                # One step for the FC solution with proper coupling 
                ξFC            = exp.(-λF*ΔtF).*ξF1 + ξF2; 
                UFC            = driftLapFC.*UFC + driftReactionFC.*f(UFC) + (1 .+αF.*g(UFC)).*ξFC;
            end

            if JC < JF && NC < NF
                # One step for the CC solution 
                ξCC            = exp.(-λC*ΔtF).*ξC1 + ξC2; 
                UCC            = driftLapCC.*UCC + driftReactionCC.*f(UCC) + (1 .+αC.*g(UCC)).*ξCC;
            end
        end
                                                                        
        # embedd UC modes into the Galerkin space of UF
        # println((length(UFF), length(UFC), length(UCF), length(UCC)))
        if method ==1
            if JC < JF && NC < NF
                ΔU                = doubleDifference(UFF,UFC,UCF,UCC);
            elseif JC < JF
                ΔU                = singleDifference(UFF,UFC);
            elseif NC < NF
                ΔU                = singleDifference(UFF,UCF);
            else
                ΔU = UFF;
            end
        elseif method==2
            if JC < JF
                ΔU                = singleDifference(UFF, UCC);
            else
                ΔU = UFF;
            end
        end
                
        ΔUSum             += ΔU;
        errFinalTimeSum   += sum( ΔU.^2 );

    end
    return ΔUSum, errFinalTimeSum; #, errAllTimeSum;
end





function mlmctest(pData,rngVec)

    β1, ϵ, ν, σ, τ, f, g, pName = pData;
    tol = 2. .^(-(3:10));

    
    #pseudoRefSol, costRef = mimcestimator(pData,rngVec,  tol[end]/4);
    # For this particular problem 
    pseudoRefSol = zeros(2^26);
   
    GC.gc()
    
    Nref = length(pseudoRefSol);
    #estVecMLMC = zeros(length(tol), Nref);
    costVecMLMC = zeros(length(tol));
    errorVecMLMC = zeros(length(tol));
    costVecMIMC = zeros(length(tol));
    errorVecMIMC = zeros(length(tol));
    
    for i in 1:length(tol)
        print(i)
        estU, cost    = mlmcestimator(pData,rngVec, tol[i]);
        costVecMLMC[i]  = cost;
        errorVecMLMC[i] = sum( singleDifference(pseudoRefSol,estU).^2 );
        
        estU, cost    = mimcestimator(pData,rngVec, tol[i]);
        costVecMIMC[i]  = cost;
        errorVecMIMC[i] = sum( singleDifference(pseudoRefSol,estU).^2 );
        
    end


    fig1 = Figure(fontsize=22)
    ax1 = Axis(fig1[2, 1],
               #title = L"$L^2(\Omega,H)$ error MLMC",
               xlabel = L"\varepsilon",
               xscale = log10,
               yscale = log10
               )
    
    lines!(ax1,tol, errorVecMLMC ,linewidth=2, color=:black, label=L"MLMC")
    scatterlines!(ax1,tol, errorVecMIMC ,linewidth=2, color=:black, label=L"MIMC")
    lines!(ax1,tol, 4*tol .^2 ,linewidth=2, color=:black, linestyle=:dashdot, label=L"c \varepsilon^2")
    leg1 = Legend(fig1[1, 1], ax1, orientation = :horizontal)
    fig1
    
    display(fig1)

    fig2 = Figure(fontsize=22)
    ax2 = Axis(fig2[2, 1],
               #title = L"$L^2(\Omega,H)$ error MLMC",
               xlabel = L"\varepsilon",
               xscale = log10,
               yscale = log10
               )
    if β1 == 2 # additive noise setting 
        
        if ν == 2 # γ=2 == β1
            slopeTextMLMC = L"c \log(\varepsilon)^2 \varepsilon^{-2}";
            slopeLineMLMC = 120*abs.(log2.(tol)).^2 .*tol.^(-2)
            slopeTextMIMC = L"c \varepsilon^{-2}";
            slopeLineMIMC = 50*(tol) .^(-2)
            
        elseif ν==4//3 # γ=3/2 < β1=2
            slopeTextMLMC = L"c \varepsilon^{-2.5}";
            slopeLineMLMC = 1100*tol .^(-2.5);
            slopeTextMIMC = L"c \varepsilon^{-2}";
            slopeLineMIMC = 220*(tol) .^(-2)
            
        elseif ν==1 #γ =3 < β1=2
            slopeTextMLMC = L"c \varepsilon^{-3}";
            slopeLineMLMC = 240*tol .^(-3);
            slopeTextMIMC = L"c \log(\varepsilon)^2 \varepsilon^{-2}";
            slopeLineMIMC = 100*abs.(log2.(tol)) .*tol.^(-2)
        end
    elseif β1 ==1 #multiplicative noise setting

        if ν == 2 # γ== 3/2 > β1 =1  (β1*ν >1)
            slopeTextMLMC = L"c \varepsilon^{-3} |\log(\varepsilon)|";
            slopeLineMLMC = 300 *tol.^(-3)
            slopeTextMIMC = L"c \varepsilon^{-2} \log(\varepsilon)^2";
            slopeLineMIMC = 70* log2.(tol).^2 .*(tol) .^(-2)
        elseif ν==4//3 # γ =7/8 > β1 = 1  (β1*ν >1)
            slopeTextMLMC = L"c \varepsilon^{-3.5} ";
            slopeLineMLMC = 140* tol .^(-3.5);
            slopeTextMIMC = L"c \varepsilon^{-2}  \log(\varepsilon)^2";
            slopeLineMIMC = 140* log2.(tol).^2 .*(tol) .^(-2)
        elseif ν==1 # γ =2 > β1 = 1  (β1*ν ==1)
            slopeTextMLMC = L"c \varepsilon^{-4} |\log(\varepsilon)|";
            slopeLineMLMC = 30* abs.(log2.(tol)) .* tol .^(-4);
            slopeTextMIMC = L"c \varepsilon^{-2}  \log(\varepsilon)^4";
            slopeLineMIMC = 100*(tol) .^(-2) .*log2.(tol).^4
        end

    end     

    

    
    lines!(ax2,tol, costVecMLMC ,linewidth=2, color=:black, label=L"MLMC")
    scatterlines!(ax2,tol, costVecMIMC ,linewidth=2, color=:black, label=L"MIMC")
    lines!(ax2,tol, slopeLineMLMC ,linewidth=2, color=:black, linestyle=:dash, label=slopeTextMLMC)
    lines!(ax2,tol, slopeLineMIMC,linewidth=2, color=:black, linestyle=:dashdotdot, label=slopeTextMIMC)
    leg2 = Legend(fig2[1, 1], ax2, orientation = :horizontal)
    fig2
    display(fig2)

    if isdir(pName)==0
        mkdir(pName);
    end
    folder = pName*"/";
    save(folder*"errorMultimethods.pdf", fig1)
    save(folder*"costMultimethods.pdf", fig2)

        
end



function mimcestimator(pData,rngVec, tol)

    β1, ϵ, ν, σ, τ, f,g, pName = pData

    # rates compared to Thm 2.6 we set N = 2*ceil(Int,2*2 .^(2*(0:L)/(β*ν)));
    # instead of N = 2*ceil(Int,2*2 .^((0:L)/(β*ν))) because M^{-1} instead of $M^{-1/2}
    # in J
    method =1; #method 1 == mimc
    #β1 = strong rate in time/in J 
    β2 = ν; #strong rate in space/in N -- We choose μ_j = j^{-1.01} precisely to achieve β2 = κ ν = ν. 
    α1 = β1/2;
    α2 = β2/2;
    ξ1 = α1 + (1 - β1)/10;
    ξ2 = α2 + (1 - β2)/10;
    
   
    if max(β1,β2) >1 && min(β1,β2)≥1
        L = ceil(Int, log2(1/tol));
    elseif β1==β2==1
        L = ceil(Int, log2(1/tol) + log2(log2(1/tol)));
    end
    #we do not cover last case in code 

    # possible resolution value (not all resolutions will be used/sampled from)
    Llim = ceil(Int, 2*L);
    J = 2* 2 .^(0:Llim);N = 2*ceil.(Int,2*2 .^(0:Llim));
    m = zeros(Int,Llim +1, Llim+1);

    mimcEstU = zeros(N[1]);
    costMimc = 0;
    
    for ℓ2 = 0:Llim

        if ℓ2 >0
            mimcEstU = vcat(mimcEstU, zeros(N[ℓ2+1]-N[ℓ2]));
        end
        for ℓ1 =0:Llim
            if ξ1*ℓ1 + ξ2*ℓ2 ≤ L
               
                if min(β1,β2) > 1
                    m[ℓ1+1, ℓ2+1] = ceil(Int, 24*tol^(-2)/(J[ℓ1+1]^((1+β1)/2)*N[ℓ2+1]^((1+β2)/2)));
                    
                elseif max(β1,β2) > min(β1,β2)==1
                    m[ℓ1+1, ℓ2+1] = ceil(Int, 10*L*tol^(-2)/(J[ℓ1+1]^((1+β1)/2)*N[ℓ2+1]^((1+β2)/2)));
                elseif β1==β2==1
                    m[ℓ1+1, ℓ2+1] = ceil(Int, 8*L^2*tol^(-2)/(J[ℓ1+1]^((1+β1)/2)*N[ℓ2+1]^((1+β2)/2)));
                end

                if ℓ1 + ℓ2 ==0
                    m[1, 1] = 12*m[1, 1];
                    NFine = N[1]; NCoarse = N[1]; JFine = J[1]; JCoarse = J[1];
                elseif ℓ1 ==0
                    m[ℓ1+1, ℓ2+1] = 4 * m[ℓ1+1, ℓ2+1];
                    NFine = N[ℓ2+1]; NCoarse = N[ℓ2]; JFine = J[1]; JCoarse = J[1];

                elseif ℓ2 ==0
                    m[ℓ1+1, ℓ2+1] = 8 * m[ℓ1+1, ℓ2+1];
                    NFine = N[1]; NCoarse = N[1]; JFine = J[ℓ1+1]; JCoarse = J[ℓ1];
                else
                    NFine = N[ℓ2+1]; NCoarse = N[ℓ2]; JFine = J[ℓ1+1]; JCoarse = J[ℓ1];
                end

                println("Mimc level:", (ℓ1,ℓ2), "with m(ℓ1,ℓ2)= ", m[ℓ1+1,ℓ2+1]);
                if ℓ2 < 20
                    procUsed = workers;
                else
                    procUsed = min(workers,10);
                end
                    
                
                Mpar =  m[ℓ1+1,ℓ2+1] ÷ procUsed
                if Mpar >0 
                    output = pmap(rng -> coupledSolvers(method, NFine, NCoarse, JFine,JCoarse, pData, rng, Mpar), rngVec[1:procUsed]) 
                    for i=1:size(output)[1]
                        mimcEstU += output[i][1]/m[ℓ1+1,ℓ2+1];
                    end
                end
                skiprngsahead(rngVec);
                
                remaining = m[ℓ1+1,ℓ2+1]-Mpar*procUsed;
                if remaining >0
                    output = pmap(rng -> coupledSolvers(method, NFine, NCoarse, JFine,JCoarse, pData, rng, 1), rngVec[1:remaining]) 
                    for i=1:size(output)[1]
                        mimcEstU += output[i][1]/m[ℓ1+1,ℓ2+1];
                    end
                    skiprngsahead(rngVec);
                end
                                
                costMimc += m[ℓ1+1,ℓ2+1]*N[ℓ2+1]*J[ℓ1+1]; 
            end
        end
    end 
        
    return mimcEstU, costMimc;   
end


function mlmcestimator(pData,rngVec, tol)

    β1,ϵ, ν, σ, τ, f, g, pName = pData

    # rates compared to Thm 2.6 we set N = 2*ceil(Int,2*2 .^(2*(0:L)/(β*ν)));
    # instead of N = 2*ceil(Int,2*2 .^((0:L)/(β*ν))) because M^{-1} instead of $M^{-1/2}
    # in J
    method =2; # method 2 == mlmc
    β2 = ν; #as β2 = κ*ν, but κ=1

    βMLMC = β1 # see how N is set, three lines below
    αMLMC  = βMLMC/2; 
    L = ceil(Int, log2(1/tol)/αMLMC);
    J = 2* 2 .^(0:L);N = 2*ceil.(Int,2*2 .^(β1*(0:L)/β2)); #divide by β1/β2 to equilibrate rate in space with rate in time
    γMLMC = 1+β1/ν;
    if γMLMC < βMLMC - 1e-10 
        M = ceil.(Int, 2*(1/tol)^2*2. .^(-(βMLMC+γMLMC)*(0:L)/2.));
    elseif abs(γMLMC - βMLMC) < 1e-10 # γMLMC == βMLMC with some slack as not easy to compare for floats
        println("γ = β for ϵ = ", tol);
        M = ceil.(Int, 3*L*(1/tol)^2*2. .^(-(βMLMC+γMLMC)*(0:L)/2.));
    else
        M = ceil.(Int, 5*(1/tol)^(2+ (γMLMC - βMLMC)/(2*αMLMC))*2. .^(-(βMLMC+γMLMC)*(0:L)/2.));
    end
    M[1] *=4;    
    
    println("Values mlmc method: L",L, "M vec ", M, "N vec ", N)
         
    mlmcEstU = zeros(N[1]);
    for ℓ=0:length(N)-1
        print(ℓ)
        if ℓ==0
            NFine = N[1]; NCoarse = N[1]; JFine = J[1]; JCoarse = J[1];
        else
            # embedd mlmcEstU in a Galerkin space with dim = N[ℓ+1]
            mlmcEstU = vcat(mlmcEstU, zeros(N[ℓ+1]-N[ℓ]));
            NFine = N[ℓ+1]; NCoarse = N[ℓ]; JFine = J[ℓ+1]; JCoarse = J[ℓ];
        end

        if ℓ < 19
            procUsed = workers;
        elseif ℓ==19 # high level samples require a lot or memory, so avoid too many parallel processors
            procUsed = min(workers, 20);
        else
            procUsed = min(workers, 10);
        end
        
        Mpar = M[ℓ+1] ÷ procUsed;
        if Mpar >0 
            output = pmap(rng -> coupledSolvers(method, NFine, NCoarse, JFine,JCoarse, pData, rng, Mpar), rngVec[1:procUsed]); 
            for i=1:size(output)[1]
                mlmcEstU += output[i][1]/M[ℓ+1];
            end
            skiprngsahead(rngVec);
        end
        remaining = M[ℓ+1] - Mpar*procUsed;
        if remaining >0
            output = pmap(rng -> coupledSolvers(method, NFine, NCoarse, JFine,JCoarse, pData, rng, 1), rngVec[1:remaining]);
            for i=1:size(output)[1]
                mlmcEstU += output[i][1]/M[ℓ+1];
            end
            skiprngsahead(rngVec);
        end
    end
    
    # Return value in a fixed-dimension Galerkin space?
    #costMlmc = sum(M.*N.*log.(N).*J);
    costMlmc = sum(M.*N.*J);
    return mlmcEstU, costMlmc;   
end






function doubleDifferenceRates(pData, rngVec,M)
    β1, ϵ, ν, σ, τ, f, g, pName = pData
    
    N = 4 * round.(Int, 2 .^(1:11)); J = 2* 2 .^(1:11);
    
    strongErrorFinalMse      = zeros( length(J)-1, length(N)-1);
    #strongErrorMaxTimeMse    = zeros(length(J)-1,length(N)-1);
    strongRateRef  = zeros(length(J)-1, length(N)-1);
    weakError      = zeros( length(J)-1, length(N)-1);  

      
    Mpar  = M ÷ workers; M = Mpar*workers;#just to be safe with MC number
    method =1;
    for ℓ₂ = 1:length(N)-1
        for ℓ₁ =1:length(J)-1
            #println((ℓ₁, ℓ₂))
            output = pmap(rng -> coupledSolvers(method,N[ℓ₂+1], N[ℓ₂], J[ℓ₁+1], J[ℓ₁], pData, rng, Mpar), rngVec)
            ΔU = zeros(N[ℓ₂+1]);
            errAllTimeTmp = zeros(J[ℓ₁]); 
            for i=1:workers
                ΔU                  += output[i][1]/M;
                strongErrorFinalMse[ℓ₁, ℓ₂]  += output[i][2]/M;
                #errAllTimeTmp += output[i][3]/M;
            end
            #strongErrorMaxTimeMse[ℓ₁, ℓ₂]  = maximum(errAllTimeTmp);
            weakError[ℓ₁, ℓ₂]      = sum((ΔU).^2);
            strongRateRef[ℓ₁, ℓ₂]  = 1/((N[ℓ₂])*J[ℓ₁]);
        end
    end

    strongErrorFinalL2    = sqrt.(strongErrorFinalMse);
    weakError             = sqrt.(weakError);

    # Long ugly code for plotting that could be shortened
        
    sol, logFitPlane= fitPlane(J,N,ν,strongErrorFinalL2);
    sol2, logFitPlane2 = fitPlane2(J,N,ν,strongErrorFinalL2);
    climits = extrema( [log2.(strongErrorFinalL2); logFitPlane]);
    tv = round(Int64,climits[1])-1:1:round(Int64, climits[2])+2;
    fig1 = Figure(fontsize=25, resolution=(1200,570))
    ax1,hm1 = contourf(fig1[1,1], (1:length(J)-1),(1:length(N)-1),  log2.(strongErrorFinalL2), levels = tv, colormap = :Spectral);
    ax1.xlabel = L"\ell_1";
    ax1.ylabel=L"\ell_2";
    ax1.xticks = 1:2:length(J)-1; 
    ax1.yticks = 1:2:length(N)-1; 
    ax1.title  = L"\log_2 e_F(\ell_1,\ell_2)"
    limits!(ax1, 1,length(J)-1, 1, length(N)-1);

    
    
    ax2,hm2 = contourf(fig1[1,2], (1:length(J)-1),(1:length(N)-1),  min.(logFitPlane,logFitPlane2),  levels = tv, colormap = :Spectral);
    ax2.xticks = 1:2:length(J)-1; 
    ax2.yticks = 1:2:length(N)-1;
    ax2.xlabel = L"\ell_1";
    
    constString = "+c_1"; #constString = sol[3] >=0 ? "+"*string(sol[3]) : string(sol[3]);
    plane1String = string(sol[1])*"\\ell_1 "*string(sol[2])*"\\ell_2"*constString;
    constString2 = "+c_2";   # constString2 = sol2[2] >=0 ? "+"*string(sol2[2]) : string(sol[2]);
    plane2String = string(sol2[1])*"\\ell_2 "*constString2;
    
    axTitle2 = "\\log_2p(\\ell_1,\\ell_2) = \\min("*plane1String*" , "*plane2String*")" 
    ax2.title  = latexstring(axTitle2);
    #ax2.aspect  = AxisAspect(1);
    #ax2.ylabel=L"\ell_2";
    
    limits!(ax2, 1,length(J)-1, 1, length(N)-1);
    linkyaxes!(ax1, ax2)
    hideydecorations!(ax2, ticks = false)
    
    # hm_sublayout = GridLayout();
    # colsize!(fig1.layout,1,Aspect(0.5,1))
    # fig[1:2,1] = hm_sublayout; 
    fig1[1,3] = Colorbar(fig1, hm1, ticks=tv[1:2:end]);
    
    trim!(fig1.layout)
    display(fig1)


    fig1b = Figure(fontsize=25, resolution=(1200,570))
    ax1b,hm1b = contourf(fig1b[1,1], (1:length(J)-1),(1:length(N)-1),  log2.(strongErrorFinalL2), levels = tv, colormap = :Spectral);
    ax1b.xlabel = L"\ell_1";
    ax1b.ylabel=L"\ell_2";
    ax1b.xticks = 1:2:length(J)-1; 
    ax1b.yticks = 1:2:length(N)-1; 
    ax1b.title  = L"\log_2 e_F(\ell_1,\ell_2)"
    limits!(ax1b, 1,length(J)-1, 1, length(N)-1);

    
    
    ax2b,hm2b = contourf(fig1b[1,2], (1:length(J)-1),(1:length(N)-1),  logFitPlane,  levels = tv, colormap = :Spectral);
    ax2b.xticks = 1:2:length(J)-1; 
    ax2b.yticks = 1:2:length(N)-1;
    ax2b.xlabel = L"\ell_1";
    ax2bTitle = "\\log_2p(\\ell_1,\\ell_2) =" *plane1String; 
    ax2b.title  = latexstring(ax2bTitle);
    
    limits!(ax2b, 1,length(J)-1, 1, length(N)-1);
    linkyaxes!(ax1b, ax2b)
    hideydecorations!(ax2b, ticks = false)
    
   
    fig1b[1,3] = Colorbar(fig1b, hm1b, ticks=tv[1:2:end]);
    
    trim!(fig1b.layout)
    display(fig1b)
   
    
    fig2 = Figure(fontsize=25, resolution=(1200,570))
    sol, logFitPlane = fitPlane(J,N,ν,weakError);
    sol2, logFitPlane2 = fitPlane2(J,N,ν,weakError);
    climits = extrema( [log2.(weakError); logFitPlane]);
    tv = round(Int64,climits[1])-1:1:round(Int64, climits[2])+2;
   

    ax3, hm3 = contourf(fig2[1,1], (1:length(J)-1),(1:length(N)-1),  log2.(weakError), levels = tv, colormap = :Spectral);
    ax3.xlabel = L"\ell_1";
    ax3.ylabel=L"\ell_2";
    ax3.xticks = 1:2:length(J)-1; 
    ax3.yticks = 1:2:length(N)-1; 
    ax3.title  = L"\log_2 e_W(\ell_1,\ell_2)"
    limits!(ax3, 1,length(J)-1, 1, length(N)-1);

    
    ax4,hm4 = contourf(fig2[1,2], (1:length(J)-1),(1:length(N)-1),  min.(logFitPlane,logFitPlane2), levels = tv, colormap = :Spectral);
    ax4.xticks = 1:2:length(J)-1; 
    ax4.yticks = 1:2:length(N)-1;
    ax4.xlabel = L"\ell_1";

    constString = "+c_1"; #constString = sol[3] >=0 ? "+"*string(sol[3]) : string(sol[3]);
    plane1String = string(sol[1])*"\\ell_1 "*string(sol[2])*"\\ell_2"*constString;
    constString2 = "+c_2";   # constString2 = sol2[2] >=0 ? "+"*string(sol2[2]) : string(sol[2]);
    plane2String = string(sol2[1])*"\\ell_2 "*constString2;
    
    axTitle4 = "\\log_2p(\\ell_1,\\ell_2) = \\min("*plane1String*" , "*plane2String*")" 
    ax4.title  = latexstring(axTitle4);
   
    limits!(ax4, 1,length(J)-1, 1, length(N)-1);
    linkyaxes!(ax3, ax4)
    hideydecorations!(ax4, ticks = false)
    
    
    fig2[1,3] = Colorbar(fig2, hm4, ticks=tv[1:2:end]);
    trim!(fig2.layout)
    fig2

    if isdir(pName)==0
        mkdir(pName);
    end
    folder = pName*"/";
    save(folder*"rmseDoubleDTwoPlanes.pdf", fig1.scene)
    save(folder*"rmseDoubleD.pdf", fig1b.scene)
    save(folder*"weakDoubleD.pdf", fig2.scene)


    # end long ugly code for plotting
end

function fitPlane(J,N, ν, errorFunc)

    #display(plt1);
    #Least-square fit of plane to strongError/weakError/maxError
    #log2(plane(ℓ_1, ℓ_2)) = -\beta_1 ℓ_1 - \beta_2 \ell_2 + c
    # for "asymptotic values" of l_1 and l_2 for l_1, l_2 \ge 5.
    l1Coord = zeros(0); l2Coord = zeros(0); errorCoord = zeros(0);
       
    for ℓ2 = 1:length(N)-1
        for ℓ1 = ceil(Int, ν*ℓ2)+3:length(J)-1
            append!(l1Coord, ℓ1);
            append!(l2Coord, ℓ2);
            append!(errorCoord, log2(errorFunc[ℓ1,ℓ2]));
        end
    end
       
    A = [l1Coord l2Coord ones(length(l2Coord))];
    sol = A\errorCoord; # sol = [-β_1 -β_2 c]
    println("Rates beta1/alpha1 and beta2/alpha1: ", sol[1:2])
        
    l2 = ones(length(J)-1).*(1:length(N)-1)'; 
    l1 = (1:length(J)-1) .*ones(length(N)-1)';
    logFitPlane = sol[1]*l1 .+ sol[2]*l2 .+ sol[3];     
    sol =trunc.(sol,digits=2);

        
    return sol, logFitPlane; 
    
end

function fitPlane2(J,N, ν, errorFunc)

    #display(plt1);
    #Least-square fit of plane to strongError/weakError/maxError
    #log2(plane(ℓ_1, ℓ_2)) = 2\beta_2 \ell_2 + c
    # for "asymptotic values" of l_1 and l_2 for l_1, l_2 \ge 5.
    l2Coord = zeros(0); errorCoord = zeros(0);

    for ℓ1 = 1:length(J)-1
        for ℓ2 = 3*ℓ1+1: length(N)-1
            append!(l2Coord, ℓ2);
            append!(errorCoord, log2(errorFunc[ℓ1,ℓ2]));
        end
    end
       
    A = [l2Coord ones(length(l2Coord))];
    sol = A\errorCoord; # sol = [-β c]
    println("Alternative rate beta/alpha: ", sol[1])
        
    l2 = ones(length(J)-1).*(1:length(N)-1)'; 
    logFitPlane = sol[1]*l2 .+ sol[2];     
    sol =trunc.(sol,digits=2);
    
    return sol, logFitPlane; 
end


function mlmcDifferenceRates(pData, rngVec,M)
    β1,ϵ, ν, σ, τ, f, g, pName = pData

    β2 = ν; #as β2 = κ*ν, but κ=1
    #N = 2 * round.(Int, 2*2 .^(β1*(1:8)/β2));
    N = 2 * round.(Int, 2*2 .^((1:8)));
    J = 2* 2 .^(1:8);

    #print("N vector:", N)
    
    
    strongErrorFinalMse    = zeros( length(J));
    #strongErrorMaxTimeMse  = zeros(length(J));
    weakError      = zeros(length(J));  

      
    Mpar  = M ÷ workers;
    M     = Mpar*workers;#just to be safe with MC number
    ΔU = [];
    method =2;
    for ℓ = 1:length(J)
        #println(ℓ)
        if ℓ==1
            output = pmap(rng -> coupledSolvers(method, N[ℓ], N[ℓ], J[ℓ], J[ℓ], pData, rng, Mpar), rngVec);
            #errAllTimeTmp = zeros(J[ℓ]); 
        else
            output = pmap(rng -> coupledSolvers(method, N[ℓ], N[ℓ-1], J[ℓ], J[ℓ-1], pData, rng, Mpar), rngVec);
            #errAllTimeTmp = zeros(J[ℓ-1]); 
        end
        ΔU = zeros(N[ℓ]);

        for i=1:workers
            ΔU                  += output[i][1]/M;
            strongErrorFinalMse[ℓ]  += output[i][2]/M;
            #errAllTimeTmp += output[i][3]/M;
        end
        #strongErrorMaxTimeMse[ℓ]  = maximum(errAllTimeTmp);
        weakError[ℓ]      = sum((ΔU).^2);
    end
    
    #strongErrorMaxTimeMse = sqrt.(strongErrorMaxTimeMse);
    strongErrorFinalL2   = sqrt.(strongErrorFinalMse);
    weakError           = sqrt.(weakError);

    #print(strongErrorMaxTimeMse-strongErrorFinalL2)
    sol, logFitLine = fitLine(J,ν,strongErrorFinalL2);

    fig1 = Figure(fontsize=25)
    ax1 = Axis(fig1[2, 1],
               #title = L"$L^2(\Omega,H)$ error MLMC",
               xlabel = L"\ell")
    
    lines!(ax1,(0:length(J)-1), log2.(strongErrorFinalL2),linewidth=2, color=:black, label=L"\log_2 e_F(\ell)")
    labelString = "\\log_2 l(\\ell) ="*string(sol[1])*"\\ell + c";
    lines!(ax1, (0:length(J)-1), logFitLine, linewidth=2,color =:black, linestyle = :dash, label=latexstring(labelString))
    
    leg1 = Legend(fig1[1, 1], ax1, orientation = :horizontal)
    ax1.xticks = 0:length(J)
    fig1

    # sol, logFitLine = fitLine(J,ν,strongErrorMaxTimeMse);
    # fig2 = Figure(fontsize=30)
    # ax2 = Axis(fig2[2, 1],
    #            #title = L"$L^2(\Omega,H)$ error max all times MLMC",
    #            xlabel = L"\ell"               
    #            )
    
    # lines!(ax2,(0:length(J)-1), log2.(strongErrorMaxTimeMse),linewidth=2, color=:black, label=L"\log_2 e_M(\ell)")
    # labelString = "\\log_2 l(\\ell) ="*string(sol[1])*"\\ell + c";
    # lines!(ax2, (0:length(J)-1), logFitLine, linewidth=2,color =:black, linestyle = :dash, label=latexstring(labelString))
    # leg2 = Legend(fig2[1, 1], ax2, orientation = :horizontal)

    # fig2

    sol, logFitLine = fitLine(J,ν,weakError);
    fig3 = Figure(fontsize=30)
    ax3 = Axis(fig3[2, 1],
               #title = "Weak error MLMC",
               xlabel = L"\ell"
               )
    
    lines!(ax3,(0:length(J)-1), log2.(weakError),linewidth=2, color=:black, label=L"\log_2 e_W(\ell)")
    labelString = "\\log_2 l(\\ell) ="*string(sol[1])*"\\ell + c";
    lines!(ax3, (0:length(J)-1), logFitLine, linewidth=2,color =:black, linestyle = :dash,  label=latexstring(labelString))
    leg3 = Legend(fig3[1, 1], ax3, orientation = :horizontal)
    
    fig3
    if isdir(pName)==0
        mkdir(pName);
    end
    folder = pName*"/";

    trim!(fig1.layout)
    save(folder*"rmseFinalSingleD.pdf", fig1)
    #save(folder*"rmseMaxSingleD.pdf", fig2)
    save(folder*"weakFinalSingleD.pdf", fig3)
    display(fig1)
    
end


function fitLine(J,v, errorFunc)

    #display(plt1);
    #Least-square fit of line to strongError/weakError/maxError
    #log2(line(\ell) = -\beta ℓ + c
    # for "asymptotic values" of l \ge 2.

    l = (3:length(J));
    
    A = [l ones(length(l))];
    sol = A\log2.(errorFunc[3:end]); # sol = [-β_1 c]
    print("Rates beta and constant c: ", sol[1:2])
    #println(errorFunc)

    line = (0:length(J)-1);
    logFitLine = Array(sol[1]*line .+ sol[2]);

    sol =trunc.(sol,digits=2);
    
    return sol, logFitLine; 
end

@everywhere function U₀(N, rng)
    # initial data 
    #u₀(x) = 2x .*(x .≤ 0.5) + 2*(1 .-x) .*(x .>0.5)
    # this is a sine series given below
    
    U = zeros(N);
    c = 4* sqrt(2)/pi^2;
    
    for j =1:4:N
        U[j] = c/j^2
    end
    
    for j =3:4:N
        U[j] = -c/j^2;
    end
            
    return U;
end
