%Generate data 
clear all, close all

kernels = {'{@covLINiso}'; '{@covMaterniso, 3}'; '{@covPoly, 3}'; '{@covRQiso}'; '{@covSEiso}'; '{@covPeriodic}';}; %'@covSM';};
kernels_hyp = {'l'; '[ell;sf]'; '[c;sf]'; '[ell;sf;al]'; '[ell;sf]'; '[ell;p;sf]'; '[w;m(:);v(:)]';};
methods = {'ll', 'loo'};
num_hyp = [1 2 2 3 2 3 2];
dataset_sizes = [20 50 100 500 1000];
D = 1;

num_scenarios = 50;
num_simulations = 50;
num_kernels = length(kernels);
num_sizes = length(dataset_sizes);

nlml_results = cell(num_kernels * num_sizes,3);
index = 1;
k_start = 1;
n_start = 1 + num_sizes;
scenario_start = 1 + num_scenarios;
sim_start = 1 + num_simulations;
test_kernel_start = 1 + num_kernels;

for k = k_start:num_kernels
    disp('Kernel')
    disp(k)
    nlml_results{index, 1} = kernels{k};
    
    if (n_start > length(dataset_sizes))
       n_start = 1; 
    end

    for n = dataset_sizes(n_start:length(dataset_sizes))
        disp('Size')
        nlml_results{index, 2} = n;        
           
        if (scenario_start > num_scenarios)
            scenario_i = 1;
            nlml_result_for_scenarios = cell(num_scenarios,3);
            scenario_start = 1;
        end
        
        for scenario_num = scenario_start:num_scenarios
            
            if (sim_start > num_simulations)
                %Init params
                sf = 5 * rand; 
                ell = 5 * rand; 
                p = round(8 * rand) + 2; 
                al = 5 * rand; 
                c = 5 * rand;
                l = rand(1);
                Q = 2; 
                w = ones(Q,1)/Q; 
                m = rand(D,Q); 
                v = rand(D,Q);

                meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
                covfunc = eval(kernels{k}); hyp.cov = eval(kernels_hyp{k});
                likfunc = @likGauss; sn = 1 * rand + 0.5; hyp.lik = log(sn);

                nlml_result_for_simulations = cell(num_simulations,3);
                nlmls_list_for_kernels = zeros(num_kernels, 2);
                sim_i = 1;
                sim_start = 1;
            end
            
            for sim_num = sim_start:num_simulations
                
                if (test_kernel_start > num_kernels)
                    x = randn(n,D); 
                    K = feval(covfunc{:}, hyp.cov, x) + exp(hyp.lik)*eye(n);
                    mu = feval(meanfunc{:}, hyp.mean, x);
                    y = chol(K)'*gpml_randn(0.15, n, 1) + mu;       
                    nlmls_list_for_kernels = zeros(num_kernels, 2);
                    test_kernel_start = 1;
                end
                    
                for i = test_kernel_start:num_kernels
                   test_covfunc = eval(kernels{i});               
                   temp_hyp.cov = zeros(num_hyp(i), 1); temp_hyp.mean = [0; 0]; temp_hyp.lik = log(0.1);

                   hyp_ll = minimize(temp_hyp, @gp, -100, @infExact, meanfunc, test_covfunc, likfunc, x, y);
                   nlml_ll = gp(hyp_ll, @infExact, meanfunc, test_covfunc, likfunc, x, y);
                   nlmls_list_for_kernels(i, 1) = nlml_ll;

                   hyp_loo = minimize(temp_hyp, @gp, -100, @infLOO, meanfunc, test_covfunc, likfunc, x, y);
                   nlml_loo = gp(hyp_loo, @infExact, meanfunc, test_covfunc, likfunc, x, y);
                   nlmls_list_for_kernels(i, 2) = nlml_loo;
                   
                   test_kernel_start = test_kernel_start + 1;
                   save('gp_kernel_vars.mat');
                end
                
                min_val = min(nlmls_list_for_kernels(:));
                [min_kernel_i min_method_i] = ind2sub(size(nlmls_list_for_kernels), find(nlmls_list_for_kernels == min_val));
                
                num_min = length(min_kernel_i);
                
                nlml_result_for_simulations(sim_i:(sim_i + num_min - 1), 1) = kernels(min_kernel_i);  
                nlml_result_for_simulations(sim_i:(sim_i + num_min - 1), 2) = methods(min_method_i);  
                nlml_result_for_simulations(sim_i:(sim_i + num_min - 1), 3) = cellstr(strjoin(nlml_result_for_simulations(sim_i:(sim_i + num_min - 1), 1:2), ';'));
                sim_i = sim_i + num_min;
                sim_start = sim_start + 1;
                save('gp_kernel_vars.mat');
            end
            
            pop_kernel = get_popular_cell_element(nlml_result_for_simulations(:, 1));
            num_kernel = length(pop_kernel);
            
            pop_method = get_popular_cell_element(nlml_result_for_simulations(:, 2));
            num_method = length(pop_method);
            
            pop_combo = get_popular_cell_element(nlml_result_for_simulations(:, 3));
            num_combo = length(pop_combo);
            
            nlml_result_for_scenarios(scenario_i:(scenario_i + num_kernel - 1), 1) = pop_kernel;  
            nlml_result_for_scenarios(scenario_i:(scenario_i + num_method - 1), 2) = pop_method;  
            nlml_result_for_scenarios(scenario_i:(scenario_i + num_combo - 1), 3) = pop_combo;
            scenario_i = scenario_i + max([num_kernel num_method num_combo]);
            scenario_start = scenario_start + 1;
            save('gp_kernel_vars.mat');
        end
        results = cell(1,3);
        results{1, 1} = get_popular_cell_element(nlml_result_for_scenarios(:, 1));
        results{1, 2} = get_popular_cell_element(nlml_result_for_scenarios(:, 2));
        results{1, 3} = get_popular_cell_element(nlml_result_for_scenarios(:, 3));
                
        nlml_results{index, 3} = results;
        
        index = index + 1;
        n_start = n_start + 1;
        save('gp_kernel_vars.mat');
    end
    k_start = k_start + 1;
    save('gp_kernel_vars.mat');
end
