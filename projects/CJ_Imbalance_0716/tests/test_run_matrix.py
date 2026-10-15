from experiments.run_matrix import build_planned_jobs


def test_planned_jobs_cover_both_students_and_never_enable_locked_test():
    jobs = build_planned_jobs()

    branches = {job["branch"] for job in jobs}
    assert branches == {"student_a", "student_b"}
    assert all(job["allow_locked_test"] is False for job in jobs)
    assert all(len(job["config_hash"]) == 64 for job in jobs)
    assert all(job["status"] == "planned" for job in jobs)
    assert {job["class_mode"] for job in jobs} == {"four", "six"}

