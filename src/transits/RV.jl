"""
    Abstract type for RV-like data. The first field should be the `epochs::Vector{T}` at which the measurement is made.
"""
abstract type RVOutput{T} <: AbstractOutput{T} end


struct RadialVelocities{T<:AbstractFloat} <: RVOutput{T}
    epochs::Vector{T}
    rv::Vector{T}
    drvdq0::Array{T,3}
    drvdelements::Array{T,3}

    # Internal
    nrv::Int64
    ti::Int64
    drvdq::Array{T,3}
    lasttime::Vector{T} #Maybe not useful too.
    t0::T #Not super nice but feels necessary to reset in zero_out!
    lastepoch::Vector{Int64}
    s_prior::State{T}
    s_rv::State{T}
end


"""
    RadialVelocities(epochs, ic; ti)

Constructor for [`RadialVelocities`](@ref) type.

# Arguments
- `epochs::Vector{T}` : Epochs at which the RV measurements are needed
- `ic::ElementsIC{T}` : Initial conditions for the system

## Optional
- `ti::Int64=1` : Index of the body with respect to which the RV are measured. (Default is the central body)
"""
function RadialVelocities(epochs::Vector{T},ic::ElementsIC{T},ti::Int64=1) where T<:AbstractFloat
    n = ic.nbody
    nrv = length(epochs)
    rv = zeros(T,nrv)
    drvdq0 = zeros(T,nrv,7,n)
    drvdelements = zeros(T,nrv,7,n)
    drvdq = zeros(T,1,7,n)
    lasttime = [ic.t0]
    t0 = ic.t0
    lastepoch = [1]
    s_prior = State(ic)
    s_rv = State(ic)
    return RadialVelocities(epochs,rv,drvdq0,drvdelements,nrv,ti,drvdq,lasttime,t0,lastepoch,s_prior,s_rv)
end

function zero_out!(rv::RVOutput{T}) where T
    for i in 2:length(fieldnames(typeof(rv))) #the first field of RVOutput{T} is always the epochs that shouldn't be reinitialized.
        if typeof(getfield(rv,i)) <: Array{T}
            getfield(rv,i) .= zero(T)
        end
    end
    rv.lastepoch[1] = 1
    rv.lasttime[1] = rv.t0 
end

"""

Main integrator method for Transit calculations.
"""
function (intr::Integrator)(s::State{T}, tt::TransitOutput{T},rv::RVOutput{T}, d::Derivatives{T}; grad::Bool=true) where T<:AbstractFloat

    t0 = s.t[1] # Initial time
    nsteps = abs(round(Int64,intr.tmax/intr.h))
    h = intr.h * check_step(t0, intr.tmax+t0) # get direction of integration

    for i in tt.occs
        # Compute the relative sky velocity dotted with position:
        tt.gsave[i] = g!(i,tt.ti,s.x,s.v)
    end

    istep = 0
    for _ in 1:nsteps

        # Take an integration step
        if grad
            intr.scheme(s,d,h)
        else
            intr.scheme(s,h)
        end
        istep += 1
        s.t[1] = t0 + (istep * h)



        # Check if a transit occured; record time.
        detect_transits!(s,d,tt,intr,grad=grad)
        detect_rvepochs!(s,d,rv,intr,grad=grad) #Maybe h is necessary in general if the function is called outside of step loop and to get the time direction?
    end
    # Calculate derivatives
    if grad
        calc_dtdelements!(s,tt)
        calc_drvdelements!(s,rv)
    end
end

"""Wrapper so the user doesn't need to create a `Derivatives` type."""
function (intr::Integrator)(s::State{T}, tt::TransitOutput{T},rv::RVOutput{T}; grad::Bool=true, return_arrays::Bool=false) where T<:AbstractFloat
    # Preallocate arrays
    d = Derivatives(T, s.n)

    intr(s, tt, rv, d, grad=grad)
    if return_arrays; return d; end # Return preallocated arrays
    return
end


# Includes for source
files = ["velocimetry.jl"]
include.(files)