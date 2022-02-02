import FESTIM
import fenics
import pytest
import sympy as sp
import numpy as np
from pathlib import Path
import timeit


# System tests

def test_run_temperature_stationary(tmpdir):
    '''
    Check that the temperature module works well in 1D stationary
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + 2*FESTIM.x**2
    size = 1

    my_materials = [
        FESTIM.Material(id=1, D_0=4.1e-7, E_D=0.39, thermal_cond=1)
    ]

    my_mesh = FESTIM.MeshFromRefinements(200, size=size)
    my_boundary_conditions = [
        FESTIM.DirichletBC(value=1, component=0, surfaces=[1]),
        FESTIM.DirichletBC(value=u, component="T", surfaces=[1, 2])
    ]

    my_sources = [FESTIM.Source(-4, 1, "T")]
    my_temperature = FESTIM.HeatTransferProblem(transient=False)
    my_settings = FESTIM.Settings(
        absolute_tolerance=1e10, relative_tolerance=1e-9,
        maximum_iterations=50,
        final_time=30
    )
    my_stepsize = FESTIM.Stepsize(initial_value=0.5, stepsize_change_ratio=1, dt_min=1e-5)

    my_derived_quantities = FESTIM.DerivedQuantities(file="derived_quantities.csv", folder=str(Path(d)))
    my_derived_quantities.derived_quantities = [FESTIM.TotalVolume("solute", 1)]

    my_exports = [
            FESTIM.XDMFExports(fields=['T', 'solute'], labels=["temperature", "solute"], folder=str(Path(d))),
            my_derived_quantities,
            FESTIM.Error("T", exact_solution=u)
    ]

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        boundary_conditions=my_boundary_conditions,
        sources=my_sources,
        dt=my_stepsize, settings=my_settings,
        temperature=my_temperature, exports=my_exports)
    my_sim.initialise()
    output = my_sim.run()
    assert output["error"][0] < 1e-9


def test_run_temperature_transient(tmpdir):
    '''
    Check that the temperature module works well in 1D transient
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + 2*FESTIM.x**2+FESTIM.t
    size = 1

    my_materials = FESTIM.Materials(
        [FESTIM.Material(
            id=1,
            D_0=4.1e-7, E_D=0.39,
            thermal_cond=1, rho=1, heat_capacity=1,
            borders=[0, size])
        ]
    )
    my_mesh = FESTIM.MeshFromRefinements(200, size)

    my_bcs = [
        FESTIM.DirichletBC(surfaces=[1], value=1, component=0),
        FESTIM.DirichletBC(surfaces=[1, 2], value=u, component="T")
    ]

    my_temp = FESTIM.HeatTransferProblem(transient=True, initial_value=u)

    my_sources = [
        FESTIM.Source(value=sp.diff(u, FESTIM.t) - sp.diff(u, FESTIM.x, 2), volume=1, field="T")
    ]

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        transient=True, final_time=30
    )

    my_dt = FESTIM.Stepsize(
        initial_value=0.5, stepsize_change_ratio=1,
        t_stop=40, stepsize_stop_max=0.5, dt_min=1e-5)

    my_exports = FESTIM.Exports(
        [
            FESTIM.XDMFExport("T", "temperature", str(Path(d))),
            FESTIM.Error("T", u)
        ]
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials, sources=my_sources,
        boundary_conditions=my_bcs, dt=my_dt,
        settings=my_settings, temperature=my_temp,
        exports=my_exports
    )
    my_sim.initialise()
    output = my_sim.run()

    assert output["error"][0] < 1e-9


def test_run_MMS(tmpdir):
    '''
    Test function run() for several refinements
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + sp.sin(2*fenics.pi*FESTIM.x)*FESTIM.t
    v = 1 + sp.cos(2*fenics.pi*FESTIM.x)*FESTIM.t
    size = 1
    k_0 = 2
    E_k = 1.5
    p_0 = 3
    E_p = 0.2
    T = 700 + 30*FESTIM.x
    n_trap = 1
    E_D = 0.1
    D_0 = 2
    k_B = FESTIM.k_B
    D = D_0 * sp.exp(-E_D/k_B/T)
    p = p_0 * sp.exp(-E_p/k_B/T)
    k = k_0 * sp.exp(-E_k/k_B/T)

    f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
        D * sp.diff(u, FESTIM.x, 2) - \
        sp.diff(D, FESTIM.x)*sp.diff(u, FESTIM.x)
    g = sp.diff(v, FESTIM.t) + p*v - k * u * (n_trap-v)

    def run(h):
        my_materials = FESTIM.Materials(
            [
                FESTIM.Material(id=1, D_0=D_0, E_D=E_D)
            ]
        )
        my_traps = FESTIM.Traps(
            [
                FESTIM.Trap(k_0, E_k, p_0, E_p, 1, n_trap)
            ]
        )

        my_initial_conditions = [
            FESTIM.InitialCondition(field=0, value=u),
            FESTIM.InitialCondition(field=1, value=v),
        ]

        my_mesh = FESTIM.MeshFromRefinements(round(size/h), size)

        my_bcs = [
            FESTIM.DirichletBC(surfaces=[1, 2], value=u, component=0),
            FESTIM.DirichletBC(surfaces=[1, 2], value=v, component=1),
        ]

        my_temp = FESTIM.Temperature(T)

        my_sources = [
            FESTIM.Source(f, 1, "0"),
            FESTIM.Source(g, 1, "1")
        ]

        my_settings = FESTIM.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            maximum_iterations=50,
            transient=True, final_time=0.1
        )

        my_dt = FESTIM.Stepsize(0.1/50)
        my_exports = FESTIM.Exports([
                FESTIM.XDMFExport("retention", "retention", str(Path(d))),
                FESTIM.Error(0, u),
                FESTIM.Error(1, v),
            ]
        )

        my_sim = FESTIM.Simulation(
            mesh=my_mesh, materials=my_materials, traps=my_traps,
            initial_conditions=my_initial_conditions, boundary_conditions=my_bcs,
            temperature=my_temp, sources=my_sources, settings=my_settings,
            dt=my_dt, exports=my_exports)

        my_sim.initialise()
        return my_sim.run()

    tol_u = 1e-7
    tol_v = 1e-6
    sizes = [1/1600, 1/1700]
    dt = 0.1/50
    for h in sizes:
        output = run(h)
        error_max_u = output["error"][0]
        error_max_v = output["error"][1]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v


def test_run_MMS_chemical_pot(tmpdir):
    '''
    Test function run() with conservation of chemical potential (1 material)
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + sp.sin(2*fenics.pi*FESTIM.x)*FESTIM.t + FESTIM.t
    v = 1 + sp.cos(2*fenics.pi*FESTIM.x)*FESTIM.t

    size = 1
    k_0 = 2
    E_k = 1.5
    p_0 = 3
    E_p = 0.2
    T = 700 + 30*FESTIM.x
    n_trap = 1
    E_D = 0.1
    D_0 = 2
    k_B = FESTIM.k_B
    D = D_0 * sp.exp(-E_D/k_B/T)
    p = p_0 * sp.exp(-E_p/k_B/T)
    k = k_0 * sp.exp(-E_k/k_B/T)

    f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
        D * sp.diff(u, FESTIM.x, 2) - \
        sp.diff(D, FESTIM.x)*sp.diff(u, FESTIM.x)
    g = sp.diff(v, FESTIM.t) + p*v - k * u * (n_trap-v)

    def run(h):
        my_materials = FESTIM.Materials(
            [
                FESTIM.Material(id=1, D_0=D_0, E_D=E_D, S_0=2, E_S=0.1)
            ]
        )
        my_traps = FESTIM.Traps(
            [
                FESTIM.Trap(k_0, E_k, p_0, E_p, 1, n_trap)
            ]
        )

        my_initial_conditions = [
            FESTIM.InitialCondition(field=0, value=u),
            FESTIM.InitialCondition(field=1, value=v),
        ]

        my_mesh = FESTIM.MeshFromRefinements(round(size/h), size)

        my_bcs = [
            FESTIM.DirichletBC(surfaces=[1, 2], value=u, component=0),
            FESTIM.DirichletBC(surfaces=[1, 2], value=v, component=1),
        ]

        my_temp = FESTIM.Temperature(T)

        my_sources = [
            FESTIM.Source(f, 1, "0"),
            FESTIM.Source(g, 1, "1")
        ]

        my_settings = FESTIM.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            maximum_iterations=50,
            transient=True, final_time=0.1,
            chemical_pot=True
        )

        my_dt = FESTIM.Stepsize(0.1/50)
        my_exports = FESTIM.Exports([
                FESTIM.TXTExport("solute", times=[100], label="solute", folder=str(Path(d))),
                FESTIM.Error(0, u),
                FESTIM.Error(1, v),
            ]
        )

        my_sim = FESTIM.Simulation(
            mesh=my_mesh, materials=my_materials, traps=my_traps,
            initial_conditions=my_initial_conditions, boundary_conditions=my_bcs,
            temperature=my_temp, sources=my_sources, settings=my_settings,
            dt=my_dt, exports=my_exports)

        my_sim.initialise()
        return my_sim.run()

    tol_u = 1e-7
    tol_v = 1e-6
    sizes = [1/1600]
    dt = 0.1/50
    for h in sizes:
        output = run(h)
        error_max_u = output["error"][0]
        error_max_v = output["error"][1]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert error_max_u < tol_u
        assert error_max_v < tol_v


def test_run_chemical_pot_mass_balance(tmpdir):
    '''
    Simple test checking that the mass balance in ensured when solubility
    increases.
    Creates a model with a constant concentration of mobile (c_m(t=0)=1,
    non-flux conditions at surfaces) with a varying temperature
    '''
    d = tmpdir.mkdir("Solution_Test")
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=1, E_D=0.1, S_0=2, E_S=0.1)
        ]
    )

    my_initial_conditions = [
        FESTIM.InitialCondition(field=0, value=1),
    ]

    my_mesh = FESTIM.MeshFromRefinements(5, 1)

    my_temp = FESTIM.Temperature(700 + 210*FESTIM.t)

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        transient=True, final_time=100,
        chemical_pot=True
    )

    my_dt = FESTIM.Stepsize(2)

    total_solute = FESTIM.TotalVolume("solute", 1)
    total_retention = FESTIM.TotalVolume("retention", 1)
    derived_quantities = FESTIM.DerivedQuantities()
    derived_quantities.derived_quantities = [total_solute, total_retention]
    my_exports = FESTIM.Exports([
        FESTIM.XDMFExport("retention", "retention", folder=str(Path(d))),
        derived_quantities
        ]
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        initial_conditions=my_initial_conditions,
        temperature=my_temp, settings=my_settings,
        dt=my_dt, exports=my_exports)

    my_sim.initialise()
    my_sim.run()
    assert total_solute.compute() == pytest.approx(1)
    assert total_retention.compute() == pytest.approx(1)


def test_run_MMS_soret(tmpdir):
    '''
    MMS test with soret effect
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + FESTIM.x**2 + FESTIM.t
    T = 2 + sp.cos(2*fenics.pi*FESTIM.x)*sp.cos(FESTIM.t)
    E_D = 0
    D_0 = 2
    k_B = FESTIM.k_B
    D = D_0 * sp.exp(-E_D/k_B/T)
    H = -2
    S = 3
    R = FESTIM.R
    f = sp.diff(u, FESTIM.t) - \
        sp.diff(
            (D*(sp.diff(u, FESTIM.x) +
                (H*T+S)*u/(R*T**2)*sp.diff(T, FESTIM.x))),
            FESTIM.x)
    def run(h):
        my_materials = FESTIM.Materials(
            [
                FESTIM.Material(id=1, D_0=D_0, E_D=E_D, H={"free_enthalpy": H, "entropy": S})
            ]
        )
        my_initial_conditions = [
            FESTIM.InitialCondition(field=0, value=u),
        ]

        size = 0.1
        my_mesh = FESTIM.MeshFromRefinements(round(size/h), size)

        my_source = FESTIM.Source(f, 1, "solute")

        my_temp = FESTIM.Temperature(T)

        my_bcs = [
            FESTIM.DirichletBC(surfaces=[1, 2], value=u, component=0),
        ]

        my_settings = FESTIM.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            maximum_iterations=50,
            transient=True, final_time=0.1,
            soret=True
        )

        my_dt = FESTIM.Stepsize(0.1/50)

        my_exports = FESTIM.Exports([
            FESTIM.XDMFExport("solute", "solute", folder=str(Path(d))),
            FESTIM.XDMFExport("T", "T", folder=str(Path(d))),
            FESTIM.Error("solute", u, norm="L2")
            ]
        )

        my_sim = FESTIM.Simulation(
            mesh=my_mesh, materials=my_materials,
            initial_conditions=my_initial_conditions,
            boundary_conditions=my_bcs, sources=[my_source],
            temperature=my_temp, settings=my_settings,
            dt=my_dt, exports=my_exports)

        my_sim.initialise()
        return my_sim.run()

    tol_u = 1e-7
    sizes = [1/1000, 1/2000]
    for h in sizes:
        output = run(h)
        error_max_u = output["error"][0]
        msg = 'L2 error on u is:' + str(error_max_u) + '\n \
            with h = ' + str(h)
        print(msg)
        assert error_max_u < tol_u


def test_run_MMS_steady_state(tmpdir):
    '''
    MMS test with one trap at steady state
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + FESTIM.x
    v = 1 + FESTIM.x*2
    size = 1
    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T = 700 + 30*FESTIM.x
    n_trap = 1
    E_D = 0.1
    D_0 = 2
    k_B = FESTIM.k_B
    D = D_0 * sp.exp(-E_D/k_B/T)
    p = p_0 * sp.exp(-E_p/k_B/T)
    k = k_0 * sp.exp(-E_k/k_B/T)

    f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
        D * sp.diff(u, FESTIM.x, 2) - \
        sp.diff(D, FESTIM.x)*sp.diff(u, FESTIM.x)
    g = sp.diff(v, FESTIM.t) + p*v - k * u * (n_trap-v)

    def run(h):

        my_materials = FESTIM.Materials(
            [
                FESTIM.Material(id=1, D_0=D_0, E_D=E_D)
            ]
        )

        my_trap = FESTIM.Trap(k_0, E_k, p_0, E_p, [1], n_trap)

        my_initial_conditions = [
            FESTIM.InitialCondition(field=0, value=u),
            FESTIM.InitialCondition(field=1, value=v),
        ]

        size = 0.1
        my_mesh = FESTIM.MeshFromRefinements(round(size/h), size)

        my_sources = [
            FESTIM.Source(f, 1, "solute"),
            FESTIM.Source(g, 1, "1")
        ]

        my_temp = FESTIM.Temperature(T)

        my_bcs = [
            FESTIM.DirichletBC(surfaces=[1, 2], value=u, component=0),
            FESTIM.DirichletBC(surfaces=[1, 2], value=v, component=1),
        ]

        my_settings = FESTIM.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            maximum_iterations=50,
            transient=False, final_time=0.1,
            traps_element_type="DG"
        )

        my_dt = FESTIM.Stepsize(0.1/50)

        my_exports = FESTIM.Exports([
            FESTIM.XDMFExport("solute", "solute", folder=str(Path(d))),
            FESTIM.XDMFExport("1", "1", folder=str(Path(d))),
            FESTIM.XDMFExport("retention", "retention", folder=str(Path(d))),
            FESTIM.XDMFExport("T", "T", folder=str(Path(d))),
            FESTIM.Error("solute", u),
            FESTIM.Error("1", v)
            ]
        )

        my_sim = FESTIM.Simulation(
            mesh=my_mesh, materials=my_materials, traps=my_trap,
            initial_conditions=my_initial_conditions,
            boundary_conditions=my_bcs, sources=my_sources,
            temperature=my_temp, settings=my_settings,
            dt=my_dt, exports=my_exports)

        my_sim.initialise()
        return my_sim.run()

    tol_u = 1e-10
    tol_v = 1e-7
    sizes = [1/1000, 1/2000]
    for h in sizes:
        output = run(h)
        error_max_u = output["error"][0]
        error_max_v = output["error"][1]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h)
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v


def test_chemical_pot_T_solve_stationary(tmpdir):
    """checks that the chemical potential conservation is well computed with
    type solve_stationary for temperature

    adapted to catch bug described in issue #310
    """
    d = tmpdir.mkdir("Solution_Test")
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=1, E_D=0.1, S_0=2, E_S=0.2, thermal_cond=1)
        ]
    )
    my_mesh = FESTIM.MeshFromRefinements(10, 1)

    my_temp = FESTIM.HeatTransferProblem(transient=False)
    my_bcs = [
        FESTIM.DirichletBC(surfaces=[1, 2], value=1, component="solute"),
        FESTIM.DirichletBC(surfaces=[1], value=300, component="T"),
        FESTIM.DirichletBC(surfaces=[2], value=300, component="T"),
    ]
    my_settings = FESTIM.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=20,
        chemical_pot=True,
        transient=True, final_time=100,
    )
    my_dt = FESTIM.Stepsize(10, stepsize_change_ratio=1.2, dt_min=1e-8)
    my_derived_quantities = FESTIM.DerivedQuantities()
    my_derived_quantities.derived_quantities = [
        FESTIM.TotalSurface("solute", 2)
    ]
    my_exports = FESTIM.Exports([
        FESTIM.XDMFExport("solute", "solute", folder=str(Path(d))),
        my_derived_quantities
        ]
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        boundary_conditions=my_bcs,
        temperature=my_temp, settings=my_settings,
        dt=my_dt, exports=my_exports)

    my_sim.initialise()
    out = my_sim.run()
    assert out["derived_quantities"][-1][1] == pytest.approx(1)


def test_performance_xdmf(tmpdir):
    '''
    Check that the computation time when exporting every 10 iterations to XDMF
    is reduced
    '''
    d = tmpdir.mkdir("Solution_Test")

    def init_sim(nb_exports_iter):
        my_materials = FESTIM.Materials(
            [
                FESTIM.Material(id=1, D_0=1, E_D=1)
            ]
        )
        my_mesh = FESTIM.MeshFromRefinements(200, 1)

        my_temp = FESTIM.Temperature(300)

        my_settings = FESTIM.Settings(
            absolute_tolerance=1e10,
            relative_tolerance=1e-9,
            transient=True, final_time=30,
        )
        my_dt = FESTIM.Stepsize(4)
        xdmf1 = FESTIM.XDMFExport(
            "solute", "solute", folder=str(Path(d)),
            nb_iterations_between_exports=nb_exports_iter)
        xdmf2 = FESTIM.XDMFExport(
            "retention", "retention", folder=str(Path(d)),
            nb_iterations_between_exports=nb_exports_iter)
        my_exports = FESTIM.Exports([xdmf1, xdmf2])

        my_sim = FESTIM.Simulation(
            mesh=my_mesh, materials=my_materials,
            temperature=my_temp, settings=my_settings,
            dt=my_dt, exports=my_exports)

        my_sim.initialise()
        return my_sim

    # short simulation
    start = timeit.default_timer()
    init_sim(10).run()
    stop = timeit.default_timer()
    short_time = stop - start

    # long simulation
    start = timeit.default_timer()
    init_sim(1).run()
    stop = timeit.default_timer()
    long_time = stop - start

    assert short_time < long_time


def test_performance_xdmf_last_timestep(tmpdir):
    '''
    Check that the computation time when exporting only the last timestep to
    XDMF is reduced
    '''
    d = tmpdir.mkdir("Solution_Test")

    def init_sim(export_last):
        my_materials = FESTIM.Materials(
            [
                FESTIM.Material(id=1, D_0=1, E_D=1)
            ]
        )
        my_mesh = FESTIM.MeshFromRefinements(200, 1)

        my_temp = FESTIM.Temperature(300)

        my_settings = FESTIM.Settings(
            absolute_tolerance=1e10,
            relative_tolerance=1e-9,
            transient=True, final_time=30,
        )
        my_dt = FESTIM.Stepsize(4)
        xdmf1 = FESTIM.XDMFExport(
            "solute", "solute", folder=str(Path(d)),
            last_timestep_only=export_last)
        xdmf2 = FESTIM.XDMFExport(
            "retention", "retention", folder=str(Path(d)),
            last_timestep_only=export_last)
        my_exports = FESTIM.Exports([xdmf1, xdmf2])

        my_sim = FESTIM.Simulation(
            mesh=my_mesh, materials=my_materials,
            temperature=my_temp, settings=my_settings,
            dt=my_dt, exports=my_exports)

        my_sim.initialise()
        return my_sim

    # short simulation
    start = timeit.default_timer()
    init_sim(export_last=True).run()
    stop = timeit.default_timer()
    short_time = stop - start

    # long simulation
    start = timeit.default_timer()
    init_sim(export_last=False).run()
    stop = timeit.default_timer()
    long_time = stop - start

    assert short_time < long_time


def test_export_particle_flux_with_chemical_pot(tmpdir):
    """Checks that surface particle fluxes can be computed with conservation
    of chemical potential
    """
    d = tmpdir.mkdir("Solution_Test")
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=2, E_D=1, S_0=2, E_S=1, thermal_cond=2)
        ]
    )
    my_mesh = FESTIM.MeshFromRefinements(10, 1)

    my_temp = FESTIM.Temperature(300)

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        chemical_pot=True,
        transient=False,
    )
    my_derived_quantities = FESTIM.DerivedQuantities()
    my_derived_quantities.derived_quantities = [
        FESTIM.SurfaceFlux("solute", 1),
        FESTIM.SurfaceFlux("T", 1),
        FESTIM.TotalVolume("retention", 1),
    ]
    my_exports = FESTIM.Exports([
        FESTIM.XDMFExport("solute", "solute", folder=str(Path(d))),
        my_derived_quantities
        ]
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        temperature=my_temp, settings=my_settings,
        exports=my_exports)

    my_sim.initialise()
    my_sim.run()


def test_extrinsic_trap():
    """Runs a FESTIM sim with an extrinsic trap
    """
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=2, E_D=1)
        ]
    )
    my_mesh = FESTIM.MeshFromRefinements(10, 1)

    my_traps = FESTIM.ExtrinsicTrap(
        k_0=1, E_k=0.1, p_0=1e13, E_p=0.1, materials=[1],
        form_parameters={
            "phi_0": 2.5e19,
            "n_amax": 1e-1*6.3e28,
            "f_a": 1,
            "eta_a": 6e-4,
            "n_bmax": 1e-2*6.3e28,
            "f_b": 2,
            "eta_b": 2e-4}
    )

    my_temp = FESTIM.Temperature(300)

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        final_time=1,
    )
    my_dt = FESTIM.Stepsize(0.5)

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        temperature=my_temp, settings=my_settings, traps=my_traps, dt=my_dt)

    my_sim.initialise()
    my_sim.run()


def test_steady_state_with_2_materials():
    """Runs a sim with several materials and checks that the produced value is
    not zero at the centre
    """
    # build
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=[1, 2], D_0=1, E_D=0),
            FESTIM.Material(id=3, D_0=0.25, E_D=0),
        ]
    )

    N = 16
    mesh = fenics.UnitSquareMesh(N, N)
    vm = fenics.MeshFunction("size_t", mesh, 2, 0)
    sm = fenics.MeshFunction("size_t", mesh, 1, 0)

    tol = 1E-14
    subdomain_1 = fenics.CompiledSubDomain('x[1] <= 0.5 + tol', tol=tol)
    subdomain_2 = fenics.CompiledSubDomain('x[1] >= 0.5 - tol && x[0] >= 0.5 - tol', tol=tol)
    subdomain_3 = fenics.CompiledSubDomain('x[1] >= 0.5 - tol && x[0] <= 0.5 + tol', tol=tol)
    subdomain_1.mark(vm, 1)
    subdomain_2.mark(vm, 2)
    subdomain_3.mark(vm, 3)

    surfaces = fenics.CompiledSubDomain('on_boundary')
    surfaces.mark(sm, 1)
    my_mesh = FESTIM.Mesh(mesh=mesh, volume_markers=vm, surface_markers=sm)

    my_temp = FESTIM.Temperature(30)
    my_bc = FESTIM.DirichletBC([1], value=0)
    my_source = FESTIM.Source(1, [1, 2, 3], "solute")

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=5,
        transient=False
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials, sources=[my_source],
        temperature=my_temp, settings=my_settings, boundary_conditions=[my_bc])

    # run
    my_sim.initialise()
    my_sim.run()

    # test

    assert my_sim.h_transport_problem.u(0.5, 0.5) != 0


def test_steady_state_traps_not_everywhere():
    """Creates a simulation problem with a trap not set in all subdomains runs
    the sim and check that the value is not NaN
    """
    # build
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=1, E_D=0, borders=[0, 0.25]),
            FESTIM.Material(id=2, D_0=1, E_D=0, borders=[0.25, 0.5]),
            FESTIM.Material(id=3, D_0=1, E_D=0, borders=[0.5, 1]),
        ]
    )

    my_mesh = FESTIM.MeshFromRefinements(100, 1)

    my_trap = FESTIM.Trap(1, 0, 1, 0, [1, 3], 1)

    my_temp = FESTIM.Temperature(1)
    my_bc = FESTIM.DirichletBC([1], value=1)

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=5,
        traps_element_type="DG",
        transient=False
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        traps=my_trap,
        temperature=my_temp, settings=my_settings, boundary_conditions=[my_bc])

    # run
    my_sim.initialise()
    my_sim.run()
    assert not np.isnan(my_sim.h_transport_problem.u.split()[1](0.5))


def test_no_jacobian_update():
    """Runs a transient sim and with the flag "update_jacobian" set to False.
    """

    # build
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=1, E_D=0),
        ]
    )

    my_mesh = FESTIM.MeshFromRefinements(10, 1)

    my_trap = FESTIM.Trap(1, 0, 1, 0, [1], 1)

    my_temp = FESTIM.Temperature(1)

    my_settings = FESTIM.Settings(
        final_time=10,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=5,
        update_jacobian=False
    )

    my_dt = FESTIM.Stepsize(1)

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials, dt=my_dt,
        traps=my_trap,
        temperature=my_temp, settings=my_settings)

    # run
    my_sim.initialise()
    my_sim.run()


def test_nb_iterations_bewteen_derived_quantities_compute():
    """Checks that "nb_iterations_between_compute" has an influence on the
    number of entries in derived quantities
    """
    def init_sim(nb_it_compute):
        my_materials = FESTIM.Materials(
            [
                FESTIM.Material(id=1, D_0=1, E_D=0)
            ]
        )
        my_mesh = FESTIM.MeshFromRefinements(10, 1)

        my_temp = FESTIM.Temperature(300)

        my_settings = FESTIM.Settings(
            absolute_tolerance=1e10,
            relative_tolerance=1e-9,
            final_time=30
        )

        my_dt = FESTIM.Stepsize(4)

        my_derived_quantities = FESTIM.DerivedQuantities(nb_iterations_between_compute=nb_it_compute)
        my_derived_quantities.derived_quantities = [
            FESTIM.TotalVolume("retention", 1),
        ]
        my_exports = FESTIM.Exports([
            my_derived_quantities
            ]
        )

        my_sim = FESTIM.Simulation(
            mesh=my_mesh, materials=my_materials,
            temperature=my_temp, settings=my_settings,
            exports=my_exports, dt=my_dt)

        my_sim.initialise()
        return my_sim

    short_derived_quantities = init_sim(10).run()["derived_quantities"]
    long_derived_quantities = init_sim(1).run()["derived_quantities"]

    assert len(long_derived_quantities) > len(short_derived_quantities)

# TODO need to convert this test and actually test something!!
# def test_nb_iterations_bewteen_derived_quantities_export(tmpdir):
#     """Checks that a simulation with "nb_iterations_between_exports" key for
#     derived quantities doesn't raise an error
#     """
#     d = tmpdir.mkdir("temp")
#     parameters = {
#         "materials": [
#             {
#                 "E_D": 1,
#                 "D_0": 1,
#                 "id": 1
#             }
#             ],
#         "traps": [
#             ],
#         "mesh_parameters": {
#                 "initial_number_of_cells": 200,
#                 "size": 1,
#                 "refinements": [
#                 ],
#             },
#         "boundary_conditions": [
#             ],
#         "temperature": {
#             "type": "expression",
#             "value": 300
#         },
#         "solving_parameters": {
#             "type": "solve_transient",
#             "final_time": 30,
#             "initial_stepsize": 4,
#             "newton_solver": {
#                 "absolute_tolerance": 1e10,
#                 "relative_tolerance": 1e-9,
#                 "maximum_iterations": 50,
#             }
#             },
#         "exports": {
#             "derived_quantities": {
#                 "total_volume": [{
#                     "field": 'retention',
#                     "volumes":  [1],
#                 }],
#                 "folder": str(Path(d)),
#                 "nb_iterations_between_exports": 2
#             },
#             },
#     }
#     output = FESTIM.run(parameters)


def test_error_steady_state_diverges():
    """Checks that when a sim doesn't converge in steady state, an error is
    raised
    """
    # build
    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=1, E_D=1),
        ]
    )

    my_mesh = FESTIM.MeshFromRefinements(10, 1)

    my_temp = FESTIM.Temperature(-1)

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        maximum_iterations=2,
        transient=False
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        temperature=my_temp, settings=my_settings)

    # run
    my_sim.initialise()
    with pytest.raises(ValueError) as err:
        my_sim.run()

    assert "The solver diverged" in str(err.value)
