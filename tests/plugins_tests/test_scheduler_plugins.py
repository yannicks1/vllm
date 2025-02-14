# SPDX-License-Identifier: Apache-2.0


def test_scheduler_plugins():
    # simulate workload by running an example
    import runpy

    from vllm_add_dummy_scheduler.dummy_scheduler import CustomException
    current_file = __file__
    import os
    example_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_file))),
        "examples", "offline_inference/basic.py")
    try:
        runpy.run_path(example_file)
    except CustomException as e:
        assert str(e) == "Exception raised by DummyScheduler"
