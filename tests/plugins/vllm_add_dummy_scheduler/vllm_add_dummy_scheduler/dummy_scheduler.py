# SPDX-License-Identifier: Apache-2.0

from vllm.core.scheduler import Scheduler


class CustomException(Exception):
    pass


class DummyScheduler(Scheduler):

    def schedule(self):
        raise CustomException("Exception raised by DummyScheduler")
