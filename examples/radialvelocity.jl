using NbodyGradient
using Plots

a = Elements(m=1.0)
b = Elements(m=-1e-3, P=3, t0=0, ecosω=0., esinω=0.0, I=π/2)
# c = Elements(m=1e-3, P=20, t0=5, ecosω=0., esinω=0.0, I=π/2)
ic = ElementsIC(0.0, 2, a, b)#,c)

obs_duration = 20.0
tobs = collect(0:0.01:obs_duration)

analyticRV = @. (2π*NbodyGradient.GNEWT/b.P)^(1/3)*b.m*sin(b.I)/(b.m+a.m)^(2/3)/√(1-b.e^2)*(cos(2π*((tobs-b.t0)/b.P)+π/2+b.ω)-b.e*cos(b.ω))


intr = Integrator(b.P/20, obs_duration+1)

s = State(ic)
tt = TransitTiming(obs_duration+1.0, ic)
rv = RadialVelocities(tobs,ic)

intr(s,tt,rv)

const AUdtoms = 1.7314568368055555e6

begin
    fig = plot()
    plot!(fig,tobs,AUdtoms*rv.rv,label="NBodyGradient")
    plot!(fig,tobs,AUdtoms*analyticRV,label="Analytical")

end

s.m