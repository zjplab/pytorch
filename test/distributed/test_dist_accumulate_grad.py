#!/usr/bin/env python3

from typing import Callable
import logging
from functools import partial
import logging
import os

from torch.distributed import rpc
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.dist_utils import dist_init
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
import torch
import torch.distributed.autograd as dist_autograd
import torch.nn as nn

WORLD_SIZE = 2
MASTER_RANK = 0
WORKER_RANK = 1


def init_logger():
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if "debug" in os.environ else logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger


gLogger = init_logger()


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_sync(
        rref.owner(), _call_method, args=args_tup, kwargs=kwargs
    )


def _remote_method_async(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_async(
        rref.owner(), _call_method, args=args_tup, kwargs=kwargs
    )


def get_linear():
    d_in = 1
    d_out = 1
    l = nn.Linear(d_in, d_out, bias=False)
    w = torch.ones((d_out, d_in))
    w.requires_grad_()
    l.weight.data = w
    return l


class TestDistAccumulateGrad(MultiProcessTestCase):
    rpc_backend = rpc.backend_registry.BackendType.PROCESS_GROUP
    rpc_backend_options = None

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    def worker_name(self):
        return "worker1"

    def setUp(self):
        super(TestDistAccumulateGrad, self).setUp()

        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        self._spawn_processes()

    def tearDown(self):
        super(TestDistAccumulateGrad, self).tearDown()

    @dist_init
    def _worker_process(self):
        gLogger.info(f"Running the worker process...")

    @dist_init
    def _master_process(self, run_test: Callable):
        gLogger.info(f"Running the master process...")
        run_test()

    def _do_test(self, run_test: Callable):
        if self.rank == MASTER_RANK:
            self._master_process(run_test)
        elif self.rank == WORKER_RANK:
            self._worker_process()
        else:
            raise RuntimeError(f"Unknow process rank: {self.rank}")

    def _setup_tensors(self):
        self.a_rref = rpc.remote(self.worker_name(), get_linear)

        self.b = torch.tensor([2.0])
        self.b.requires_grad_()
        self.c = torch.tensor([3.0])
        self.c.requires_grad_()

        self.x = torch.zeros((1))
         
    def _forward_backward(self, num_rpcs=1):
        with dist_autograd.context() as context_id:
            self._setup_tensors()
            a_rref, b, c, x = self.a_rref, self.b, self.c, self.x

            future_x = []
            for _ in range(num_rpcs):
                future_x.append(
                    _remote_method_async(torch.nn.Linear.forward, a_rref, b)
                )

            for x_fut in future_x:
                x = x + x_fut.wait()

            y = b * c
            z = x + y
            c_hook_called = 0
            def c_hook(grad):
                gLogger.info(f"c hook called with {grad}")
                c_hook_called += 1
            # Note that this is not a post hook!
            c.register_hook(c_hook)

            dist_autograd.backward(context_id, [z], retain_graph=True)
            self.assertEqual(1, c_hook_called)
            tensor_to_grad = dist_autograd.get_gradients(context_id)
            gLogger.info(f"Got tensor to grad map: {tensor_to_grad}")
            # Grads for 'b' and 'c'
            self.assertEqual(2, len(tensor_to_grad))
            self.assertAlmostEqual(tensor_to_grad[b], 3.0 + num_rpcs)
            
            dist_autograd.backward(context_id, [z], retain_graph=False)
            self.assertEqual(2, c_hook_called)
            tensor_to_grad = dist_autograd.get_gradients(context_id)
            gLogger.info(f"Got tensor to grad map: {tensor_to_grad}")
            # Grads for 'b' and 'c'
            self.assertEqual(2, len(tensor_to_grad))
            self.assertAlmostEqual(tensor_to_grad[b], 2 * (3.0 + num_rpcs))

    def test_single_rpc(self):
        self._do_test(partial(self._forward_backward, 1))

    def test_multiple_rpc(self):
        self._do_test(partial(self._forward_backward, 10))



if __name__ == "__main__":
    run_tests()
