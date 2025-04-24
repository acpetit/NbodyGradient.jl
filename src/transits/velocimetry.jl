function detect_rvepochs!(s::State{T},d::Derivatives{T},rv::RVOutput{T},intr::Integrator{T}; grad::Bool=true) where T<:AbstractFloat
    # Save current state as prior state
    set_state!(rv.s_prior, s)

    # Check to see if a RV measurement occured during timestep.
    # Sky is x-y plane; line of sight is z.
    # Body being measured is rv.ti, rv.epochs is the list of epochs:

    currenttime = s.t[1]
    rv.lastepoch += 1

    

    while  (rv.lastepoch ≤ rv.nrv)(rv.epoch[rv.lastepoch] ≤ currenttime)
        # Compute the radial velocity:

        rv.rv[rv.lastepoch] = s.v[3,rv.ti]

        if grad
            for k=1:7, p=1:s.n
                # Note that jac_step[(i-1)*7+k,(j-1)*7+p] is the derivative of the kth coordinate
                # of planet i with respect to the pth coordinate of planet j.
                rv.drvdq0[rv.lastepoch,k,p] = s.jac_step[(rv.ti-1)*7+6,(p-1)*7+k] 
            end
        end

        set_state!(s,rv.s_prior)
        rv.lastepoch += 1
    end
    rv.lasttime = currenttime # If we do some day with negative time step
    return
end



function calc_drvdelements!(s::State{T},rv::RVOutput{T}) where T <: AbstractFloat
    for j=1:rv.nrv
        
        # Now, multiply by the initial Jacobian to convert time derivatives to orbital elements:
        for k=1:s.n, l=1:7
            tt.drvdelements[j,l,k] = zero(T)
            for p=1:s.n, q=1:7
                tt.drvdelements[j,l,k] += tt.drvdq0[j,q,p]*s.jac_init[(p-1)*7+q,(k-1)*7+l]
            end
        end
        
    end
end