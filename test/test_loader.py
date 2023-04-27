import sys

sys.path.append(".")

from utils.loaders import PSPLoader


def test_psp_loader_rcp():
    loader = PSPLoader()
    rcp = loader.load_single("instances/psp/patterson/pat103.rcp")
    assert len(rcp["durations"]) == 3
    assert rcp["durations"][0] == rcp["durations"][1]
    assert rcp["durations"][0] == rcp["durations"][2]
    assert rcp["n_jobs"] == 51
    assert rcp["n_modes"] == 51
    assert rcp["n_resources"] == 3
    assert rcp["n_renewable_resources"] == 3
    assert rcp["n_nonrenewable_resources"] == 0
    assert rcp["n_doubly_constrained_resources"] == 0
    assert len(rcp["job_info"]) == 51
    assert rcp["job_info"][0][1] == [2, 3, 4]
    # "durations": durations,
    # "resources": resources,
    assert rcp["resource_availability"] == [14, 14, 12]
    assert rcp["max_resource_availability"] == 14
    assert rcp["max_resource_request"] == 6


def test_psp_loader_sm():
    loader = PSPLoader()
    det = loader.load_single("instances/psp/272/272.sm")
    assert len(det["durations"]) == 3
    assert det["durations"][0] == det["durations"][1]
    assert det["durations"][0] == det["durations"][2]
    unc = loader.load_single("instances/psp/272/272_unc.sm")
    assert len(unc["durations"]) == 3
    assert unc["durations"][0] == [
        [0],
        [38],
        [13],
        [73],
        [10],
        [76],
        [6],
        [80],
        [65],
        [17],
        [2],
        [77],
        [72],
        [7],
        [26],
        [51],
        [21],
        [0],
    ]
    assert unc["durations"][1] == [
        [0],
        [35],
        [12],
        [70],
        [9],
        [70],
        [5],
        [75],
        [60],
        [16],
        [1],
        [70],
        [71],
        [6],
        [20],
        [50],
        [20],
        [0],
    ]
    assert unc["durations"][2] == [
        [0],
        [40],
        [15],
        [80],
        [15],
        [90],
        [10],
        [90],
        [80],
        [20],
        [3],
        [85],
        [83],
        [10],
        [30],
        [60],
        [25],
        [0],
    ]

    loader_wb = PSPLoader(generate_bounds=[0.1, 0.2])
    unc_bounds = loader_wb.load_single("instances/psp/272/272.sm")

    modes = unc_bounds["durations"][0]
    assert modes == [
        [0],
        [38],
        [13],
        [73],
        [10],
        [76],
        [6],
        [80],
        [65],
        [17],
        [2],
        [77],
        [72],
        [7],
        [26],
        [51],
        [21],
        [0],
    ]
    mins = []
    maxs = []
    for j in range(18):
        mins.append([int(modes[j][0] * 0.9)])
        maxs.append([int(modes[j][0] * 1.2)])

    assert unc_bounds["durations"][1] == mins
    assert unc_bounds["durations"][2] == maxs
