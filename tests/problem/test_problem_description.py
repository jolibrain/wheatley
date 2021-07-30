def test_problem_description(problem_description):
    assert problem_description.affectations is None
    assert problem_description.durations is None
    assert problem_description.n_jobs == 5
    assert problem_description.n_machines == 5
