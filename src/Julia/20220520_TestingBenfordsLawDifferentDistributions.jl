
function firstdigit(x::Integer)
    iszero(x) && return x
    x = abs(x)
    y = 10^floor(Int, log10(x))
    z = div(x, y)
    return z
  end


function benford(ns)
    xs = firstdigit.(ns)
    counts = [count(==(d), xs) for d in 0:9]
    return (leading = 0:9, count = counts)
  end;

mrandn(n,m) = m .* randn(n)
mrandp(n,λ,m) = m .* rand(Poisson(λ),n)
trunc_int(x) = trunc.(Int, abs.(x))


n = 10^6
m = 10^4
λ = 10
randN = trunc_int(mrandn(n,m))
randP = trunc_int(mrandp(n,λ,m))
randU = trunc_int(rand(1:m,n))
benfordN = benford(randN)
benfordP = benford(randP)
benfordU = benford(randU)
f = Figure()
    ax = Axis(f[2,1],xlabel = "Leading digit",ylabel = "density")
    #barplot!(ax,benfordN[:leading],benfordN[:count])
    barplot!(ax,benfordP[:leading],benfordP[:count])
    #barplot!(ax,benfordU[:leading],benfordU[:count])
f



benford()

ns = trunc.(Int, 1000 .* rand(10000))
xs = firstdigit.(ns)
counts = [count(x-> x == d, xs) for d in 0:9]
(leading = 0:9, count = counts)



count(==(4), xs)


using Distributions


