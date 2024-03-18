using Pkg, LinearAlgebra, FileIO, JLD2, DelimitedFiles

#= 

This code will generate the matrices as described in Section 5.1 (Rank deficient matrices).
Subsequently, it will calculate and store the smallest singular values of the corresponding 
stochastically rounded matrices. We consider row dimensions n = 10^4, 10^5, 10^6 and column
dimensions d = 10, 10^2, 10^3. This code works for these fixed dimensions, if you want to 
test on different ones, you need to modify the n, d, k, name_1, name_2, name_3 variables
accordnigly.

------------------ function create_stack_mat(A::AbstractArray, k::Int) -------------------
This function takes as imput a matrix A and an integer k, and it returns a verticaly
stacked matrix A_new with its n*k number of rows being k copies of the n rows of A.
The scaling with the 1/sqrt(k), esnures that A, A_new have the same singular values.
------------------ function create_stack_mat(A::AbstractArray, k::Int) -------------------

------------------ function SR_fix_dec(a::Real,dec_pos::Int) -----------------------------
This function takes as input a real number a and an integer dec_pos which is the precision
p as defined in Section 5. Then, it returns the number a stochastically rounded to F^{p}.
------------------ function SR_fix_dec(a::Real,dec_pos::Int) -----------------------------

------------------ function SR_fix_dec_var(a::Real,dec_pos::Int) --------------------------
This function takes as input a real number a and an integer dec_pos which is the precision
p as defined in Section 5. Then, it returns the variance of \tilde{a} based on F^{p}.
------------------ function SR_fix_dec_var(a::Real,dec_pos::Int) --------------------------

------------------ function calc_var(A::AbstractArray, dec_pos::Int) ----------------------
This function takes as input a matrix A and an integer dec_pos which is the precision
p as defined in Section 5. Then, it returns the minimum variance of the stochastic
rounding process over all columns of A. In other words, it returns the non-normalized
value of \nu (see Section 4.2).
------------------ function calc_var(A::AbstractArray, dec_pos::Int) ----------------------

------------------ function run_SR_once(A::AbstractArray,dec_pos::AbstractArray) ----------
This function takes as input a matrix A and an arrray dec_pos which consists of several
precision p options, as defined in Section 5. In our experiments, we test on p=1:5,
(dec_pos = collect(1:5)). Then, for all the given values of p, it returns the corresponding
smallest singular value of the stochastically rounded A, and its normalized and non-normalized
values minimum variance over its columns.
------------------ function run_SR_once(A::AbstractArray,dec_pos::AbstractArray) ----------

-------- function create_store_mats_sigma_min_zero() --------------------------------------
This function takes as input an integer n (number of rows), an array d (number of columns),
and array dec_pos consists of several precision p options, as defined in Section 5. In our 
experiments, we test on p=1:5,(dec_pos = collect(1:5)). It also needs an integer k, for
creating the verticaly stacked matrices. In particular, for a given value of k, it will
also construct matrices with k*n, k*10*n number of rows respectively, for each given value
of d. The remaining three input variables, name_A_G, name_A_Gk, name_A_Gk_2 are strings
serving as names for storing variables of interest. More spesifically, for each of our three
number of rows, n, k*n, k*10*n we save the smallest singular values, the normalized and
non-normalized minimum variance for all the combinations indicated by d, dec_pos. This function 
returns nothing.
-------- function create_store_mats_sigma_min_zero() --------------------------------------

=#

function create_stack_mat(A::AbstractArray, k::Int)
    m,n = size(A); A_new = zeros(k*m,n)
    j = 1; iters = 1;

    while ( iters <= k )
        A_new[j:iters*m,:] .= A;
        iters += 1;
        j += m;
    end

    return A_new/sqrt(k);
end

function SR_fix_dec(a::Real,dec_pos::Int)
    @assert(dec_pos >0,"The decimal position should be >0!");
    a_shift = a*10^dec_pos;
    fl_a = floor(a_shift);

    pr = a_shift - fl_a;

    if pr >= rand(1)[1]
        fl_a += 1;
    end

    return fl_a*10.0^(-dec_pos);

end

function SR_fix_dec_var(a::Real,dec_pos::Int)
    @assert(dec_pos >0,"The decimal position should be >0!");
    a_shift = a*10^dec_pos;
    fl_a = floor(a_shift);

    pr = a_shift - fl_a;


    return 10.0^(-2*dec_pos)*pr*(1-pr);

end

function calc_var(A::AbstractArray, dec_pos::Int)
    m,n = size(A);  s = zeros(n);
    
    for j=1:n
        s[j] = 0;
        for i=1:m
            s[j] += SR_fix_dec_var(A[i,j],dec_pos);
        end
    end
    
    return minimum(s);
    
end

function run_SR_once(A::AbstractArray,dec_pos::AbstractArray) 
    m,n = size(A); A_sr = zeros(m,n); s = zeros(length(dec_pos)); 
    v = zeros(length(dec_pos)); norm_v = copy(v);
    for h in eachindex(dec_pos)
        for i=1:m
            for j=1:n
                A_sr[i,j] = SR_fix_dec(A[i,j],dec_pos[h]);
            end
        end
        
        F = svd(A_sr); S = F.S;
        s[h] = S[end];
        v[h] = calc_var(A, dec_pos[h]);
        R = maximum(broadcast(abs, A_sr-A));
        norm_v[h] = (v[h])/(m*R^2);
    end
    
    return s, v, norm_v;
end

function create_store_mats_sigma_min_zero(n::Int, d::AbstractArray, dec_pos::AbstractArray, name_A_G::String, name_A_Gk::String, 
    name_A_Gk_2::String, k::Int)
    l = length(d); sigma_min = 0; g = length(dec_pos); S_all = -ones(l,1+3*g); V_all = -ones(l, 3*g); V_norm_all = -ones(l, 3*g);


    for j=1:l
        A_G = randn(n, d[j]);  F = svd(A_G); S_G = F.S;
        max_A_G = maximum(broadcast(abs,A_G));
        
        if S_G[end] > sigma_min
            S_G[end] = sigma_min*max_A_G;
        end
        A_G = F.U*Diagonal(S_G)*F.Vt;
        A_G = A_G/max_A_G;

        while ( maximum(A_G) > 1 || minimum(A_G) < -1)
            A_G = randn(n, d[j]);  F = svd(A_G); S_G = F.S;
            max_A_G = maximum(broadcast(abs,A_G));
            
            if S_G[end] > sigma_min
                S_G[end] = sigma_min*max_A_G;
            end
            A_G = F.U*Diagonal(S_G)*F.Vt;
            A_G = A_G/max_A_G;
        end

        name = name_A_G*name_A_Gk*name_A_Gk_2;
        name_1 = name*string(d[j])*"_0";
        
        FileIO.save(name_1*".jld2", "A", A_G);
        writedlm("max_min_"*name_1*".txt",[maximum(A_G) minimum(A_G)]);
    
        S_all[j,1] = S_G[end]/max_A_G;
        S_all[j,2:g+1], V_all[j,1:g], V_norm_all[j,1:g] = run_SR_once(A_G,dec_pos);

        FileIO.save(name*"0_all_sigmas.jld2", "S_all", S_all);
        writedlm(name*"0_all_sigmas.txt", S_all);

        FileIO.save(name*"0_all_vars.jld2", "V_all", V_all);
        writedlm(name*"0_all_vars.txt",V_all);

        FileIO.save(name*"0_all_vars_norm.jld2", "V_norm_all", V_norm_all);
        writedlm(name*"0_all_vars_norm.txt",V_norm_all);

        # This is for n = 10^6, k=100
        S_all[j,g+2:2*g+1], V_all[j, g+1:2*g], V_norm_all[j,g+1:2*g] = run_SR_once(create_stack_mat(A_G,k), dec_pos);

        FileIO.save(name*"0_all_sigmas.jld2", "S_all", S_all);
        writedlm(name*"0_all_sigmas.txt", S_all);

        FileIO.save(name*"0_all_vars.jld2", "V_all", V_all);
        writedlm(name*"0_all_vars.txt",V_all);

        FileIO.save(name*"0_all_vars_norm.jld2", "V_norm_all", V_norm_all);
        writedlm(name*"0_all_vars_norm.txt",V_norm_all);

        # This is for n = 10^7, k=1000
        S_all[j,2*g+2:end], V_all[j, 2*g+1:end], V_norm_all[j,2*g+1:end] = run_SR_once(create_stack_mat(A_G,k*10), dec_pos);

        FileIO.save(name*"0_all_sigmas.jld2", "S_all", S_all);
        writedlm(name*"0_all_sigmas.txt", S_all);

        FileIO.save(name*"0_all_vars.jld2", "V_all", V_all);
        writedlm(name*"0_all_vars.txt",V_all);

        FileIO.save(name*"0_all_vars_norm.jld2", "V_norm_all", V_norm_all);
        writedlm(name*"0_all_vars_norm.txt",V_norm_all);

    end
    
    return nothing;

end


d = [10, 100, 1000]; n = 10^4;

dec_pos = collect(1:5); k = 10;

name_1 = "randn_10e4_"; name_2 = "10e5_";
name_3 = "10e6_";

if !isdir("data_results")
    mkdir("data_results")
end

cd("data_results"); 

create_store_mats_sigma_min_zero(n, d, dec_pos, name_1, name_2, name_3, k);

cd("..");

