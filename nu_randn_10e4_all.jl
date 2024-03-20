using Pkg, LinearAlgebra, FileIO, JLD2, DelimitedFiles, Random, StatsPlots, LaTeXStrings

#= 

This code will generate the matrices as described in Section 5.2 (Rank deficient matrices
with controlled \nu).
Subsequently, it will calculate and store the smallest singular values of the corresponding 
stochastically rounded matrices. We consider row dimensions n = 10^4 and column
dimensions d = 10, 10^2, 10^3. This code works for these fixed dimensions, if you want to 
test on different ones, you need to modify the n, d, k, name_1,variables accordnigly.



------------------ function change_decimal(a::Real, dec_pos::Int, r::Int) ----------------
This function takes as input a real number a, an integer dec_pos which is the precision
p as defined in Section 5 and and an integer r. Then, it outputs two numbers a_low, a
corresponding to the elements of A^{l}, A^{h} respectively (see Section 5.2). For the
high variance element a, we set the (dec_pos + 1)st decimal digit to 5 and every 
subsequent decimal digit to 0. On the other hand, for the low variance element
we set the (dec_pos + 1)st decimal digit to either 1 or 9 (with equal probability). Then,
we also set the subsequent r-1 decimal digits to the vaue of the (dec_pos + 1)st. The larger
the value of r, the smallest the variance of a_low. 
------------------ function change_decimal(a::Real, dec_pos::Int, r::Int) ----------------

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


------------------ function run_SR_once(A::AbstractArray,dec_pos::AbstractArray, r::Int) --
This function takes as input a matrix A, an arrray dec_pos which consists of several
precision p options, as defined in Section 5 and an integer r, which is going to be part of 
the input for the function change_decimal (see above). In our experiments, we test on p=1:5,
(dec_pos = collect(1:5)). Then, for all the given values of p, it returns the corresponding
smallest singular value of the stochastically rounded matrices A^{h}, A^{l}, 
and its normalized and non-normalized values minimum variance over its columns.
------------------ function run_SR_once(A::AbstractArray,dec_pos::AbstractArray, r::Int) --

-------- function create_store_mats_sigma_min() --------------------------------------
This function takes as input an integer n (number of rows), an array d (number of columns),
and array dec_pos consists of several precision p options, as defined in Section 5. In our 
experiments, we test on p=1:5,(dec_pos = collect(1:5)). It also needs an integer r, for
creating the A^{l} matrices using the change_decimal function. The remaining input variable, 
name_A_G is a string serving as the backbone of the names for storing variables of interest. 
More spesifically, for each size, and precision combination we save the smallest singular values, 
the normalized and non-normalized minimum variance for all the combinations indicated by d, dec_pos. 
This function returns nothing.
-------- function create_store_mats_sigma_min() --------------------------------------


-------- function make_plots() -------------------------------------------------------
This function takes as input two arrays S_1, V which represent the smallest singular
values and the respective bounds (see equation 1.2). It also requires a vector of
strings representing the number of columns, the array p with the precision options
(see Section 5), and a name string for storing the plots. Subsequently, for each different
value of p, it stores a separate pdf. Fially, all the separate pdfs are being merged
and stored to one pdf file.
-------- function make_plots() -------------------------------------------------------


=#

function change_decimal(a::Real, dec_pos::Int, r::Int)
    @assert(dec_pos >0,"The decimal position should be >0!");
    str_a = collect(string(a)); j = findall(x->(x=='.'),str_a);
    str_a_low = copy(str_a);
    f_low = [1 9]; 
    f = 5; a_low = copy(a);
    if !isempty(j)
        if j[1]+dec_pos+1 <= length(str_a)
            ff = shuffle(collect(1:2))[1];
            str_a_low[j[1]+dec_pos+1] = string(f_low[ff])[1];
            str_a[j[1]+dec_pos+1] = string(f)[1];
            new_string = String[]; push!(new_string, String(str_a_low[1:j[1]+dec_pos+1]))
            for j=1:r-1
                push!(new_string, string(f_low[ff]));
            end
            a_low = parse(Float64,join(new_string));
            a = parse(Float64,join(str_a[1:j[1]+dec_pos+1]));
        end
    end
    

    return a, a_low;

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

function run_SR_once(A::AbstractArray,dec_pos::AbstractArray, r::Int) 
    m,n = size(A); s = zeros(length(dec_pos)*4); 
    v = zeros(length(dec_pos)*2); norm_v = zeros(length(dec_pos)*4); 
    R = zeros(length(dec_pos)*4);
    
    A_high = zeros(m,n); A_sr_high = zeros(m,n);
    A_sr_low = zeros(m,n); A_low = zeros(m,n);
    for h in eachindex(dec_pos)
        for i=1:m
            for j=1:n
                A_high[i,j], A_low[i,j] = change_decimal(A[i,j], dec_pos[h],r);
            end
        end

        A_high[:,2] = A_high[:,1];
        A_low[:,2] = A_low[:,1];

        for i=1:m
            for j=1:n
                A_sr_high[i,j] = SR_fix_dec(A_high[i,j],dec_pos[h]);
                A_sr_low[i,j] = SR_fix_dec(A_low[i,j],dec_pos[h]);
            end
        end
        
        F = svd(A_high); s[h] = F.S[end];
        F = svd(A_sr_high); 
        s[h+length(dec_pos)] = F.S[end];

        F = svd(A_low); s[h+2*length(dec_pos)] = F.S[end];
        F = svd(A_sr_low); s[h+3*length(dec_pos)] = F.S[end];

        v[h] = calc_var(A_high, dec_pos[h]);
        
        R_high = maximum(broadcast(abs, A_sr_high-A_high));
        R_high_th = 5*10.0^(-dec_pos[h]-1);
        R[h] = R_high_th;
        R[h+length(dec_pos)] = R_high;
        norm_v[h] = (v[h])/(m*R_high_th^2);
        norm_v[h+length(dec_pos)] = (v[h])/(m*R_high^2);

        v[h+length(dec_pos)] = calc_var(A_low, dec_pos[h]);
        R_low_th = (10^r-1)*10.0^(-dec_pos[h]-1);
        R[h+2*length(dec_pos)] = R_low_th;
        R_low = maximum(broadcast(abs, A_sr_low-A_low));
        R[h+3*length(dec_pos)] = R_low;
        norm_v[h+2*length(dec_pos)] = (v[h+length(dec_pos)])/(m*R_low_th^2);
        norm_v[h+3*length(dec_pos)] = (v[h+length(dec_pos)])/(m*R_low^2);
    end
    
    return s, v, norm_v, R;
end

function create_store_mats_sigma_min_zero(n::Int, d::AbstractArray, dec_pos::AbstractArray, name_A_G::String, r::Int)
    l = length(d);  g = length(dec_pos); S_all = -ones(l,4*g); V_all = -ones(l, 2*g); V_norm_all = -ones(l, 4*g); R_all = -ones(l,4*g);

    for j=1:l
        A_G = randn(n, d[j]);  
        
        max_A_G = maximum(broadcast(abs,A_G));
        A_G = A_G/max_A_G;
        name_1 = name_A_G*string(d[j]);
        
        FileIO.save(name_1*".jld2", "A", A_G);

        S_all[j,1:end], V_all[j,1:end], V_norm_all[j,1:end], R_all[j,1:end] = run_SR_once(A_G,dec_pos,r);

        FileIO.save(name_A_G*"0_all_sigmas.jld2", "S_all", S_all);
        writedlm(name_A_G*"0_all_sigmas.txt", S_all);

        FileIO.save(name_A_G*"0_all_vars.jld2", "V_all", V_all);
        writedlm(name_A_G*"0_all_vars.txt",V_all);

        FileIO.save(name_A_G*"0_all_vars_norm.jld2", "V_norm_all", V_norm_all);
        writedlm(name_A_G*"0_all_vars_norm.txt",V_norm_all);

        FileIO.save(name_A_G*"0_all_R.jld2", "R_all", R_all);
        
    end
    
    return nothing;

end

function make_plots(S_1::AbstractArray, V::AbstractArray, string_d::Vector{String}, p::AbstractArray, name_plot::String)
    k = length(p);
    pdf_files = String[];
    for i in eachindex(p)
        S = zeros(3,2);
        S[:,1] = S_1[:,i+k];
        S[:,2] = S_1[:,i+3*k];

         
        if i == 1

            groupedbar(string_d, S,  bar_width=0.7, fillalpha = 0.5, bar_position = :dodge, yaxis = :log10,
            label = [L"\sigma_{\min}(\tilde{A}^{h}), \, \nu \approx 1" L"\sigma_{\min}(\tilde{A}^{l}), \, \nu \approx 5 \times 10^{-4}"],
            legend = :bottomright);
            scatter!([0.325,1.325,2.325], sqrt.(V[:,i]), markersize = 10, color = 1, markershape = :star, 
            label = L"\mathcal{R}\sqrt{n \nu}, \, \nu \approx 1");
            scatter!([0.675,1.675,2.675], sqrt.(V[:,i+k]), markersize = 10, color = 2, markershape = :star, 
            label = L"\mathcal{R}\sqrt{n \nu}, \, \nu \approx 5 \times 10^{-4}");
            title!(L"p = %$i");
            savefig(name_plot*string(i)*".pdf");
            push!(pdf_files, name_plot*string(i)*".pdf");

        else
            groupedbar(string_d, S,  bar_width=0.7, fillalpha = 0.5, legend = false, bar_position = :dodge, yaxis = :log10);
            scatter!([0.325,1.325,2.325], sqrt.(V[:,i]), markersize = 10, color = 1, markershape = :star);
            scatter!([0.675,1.675,2.675], sqrt.(V[:,i+k]), markersize = 10, color = 2, markershape = :star);
            title!(L"p = %$i");
            savefig(name_plot*string(i)*".pdf");
            push!(pdf_files, name_plot*string(i)*".pdf");
        end


    end

    output_file = name_plot*".pdf";
    cmd = `pdftk $pdf_files cat output $output_file`
    run(cmd);
    return nothing;
end

# if you want to try different sizes, modify variables d and n accordnigly.
# keep in mind that those experiments are meant for tall-and-thin matrices
# so n should be >= d for each size combination.
d = [10, 100, 1000]; n = 10^4;
dec_pos = collect(1:5); 


name_1 = "nu_randn_10e4_"; 

if !isdir("data_results")
    mkdir("data_results")
end

cd("data_results"); 

r = 2; # r-1 number of repeating 9s or 1s in A_low for the probabilities
create_store_mats_sigma_min_zero(n, d, dec_pos, name_1,r);

S_all = FileIO.load(name_1*"0_all_sigmas.jld2", "S_all");
V_all = FileIO.load(name_1*"0_all_vars.jld2", "V_all");

# call function for plotting
make_plots(S_all, V_all, string.(d), dec_pos, name_1); 

cd("..")
