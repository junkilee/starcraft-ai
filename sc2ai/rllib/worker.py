from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import argparse
import sys

import ray.services as services
import ray.utils
import redis

from ray.tune.config_parser import make_parser

logger = logging.getLogger(__name__)

def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a Starcraft II reinforcement learning agent.",
        epilog="")

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--redis-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
             "of starting a new one.",
        required=True)
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--temp-dir",
        default=None,
        type=str,
        help="Manually specify the root temporary dir of the Ray process")

    return parser

def check_no_existing_redis_clients(node_ip_address, redis_client):
    # The client table prefix must be kept in sync with the file
    # "src/ray/gcs/redis_module/ray_redis_module.cc" where it is defined.
    REDIS_CLIENT_TABLE_PREFIX = "CL:"
    client_keys = redis_client.keys("{}*".format(REDIS_CLIENT_TABLE_PREFIX))
    # Filter to clients on the same node and do some basic checking.
    for key in client_keys:
        info = redis_client.hgetall(key)
        assert b"ray_client_id" in info
        assert b"node_ip_address" in info
        assert b"client_type" in info
        assert b"deleted" in info
        # Clients that ran on the same node but that are marked dead can be
        # ignored.
        deleted = info[b"deleted"]
        deleted = bool(int(deleted))
        if deleted:
            continue

        if ray.utils.decode(info[b"node_ip_address"]) == node_ip_address:
            raise Exception("This Redis instance is already connected to "
                            "clients with this IP address.")

def run(args, parser):
    redis_address = services.address_to_ip(args.redis_address)
    redirect_output = True
    ray_params = ray.parameter.RayParams(
        node_ip_address=None,
        object_manager_port=None,
        node_manager_port=None,
        object_store_memory=None,
        redis_password=None,
        num_cpus=args.ray_num_cpus,
        num_gpus=None,
        resources=None,
        plasma_directory=None,
        huge_pages=None,
        plasma_store_socket_name=None,
        raylet_socket_name=None,
        temp_dir=args.temp_dir,
        include_java=None,
        java_worker_options=None,
        load_code_from_local=None,
        _internal_config=None)

    redis_ip_address, redis_port = redis_address.split(":")

    # Wait for the Redis server to be started. And throw an exception if we
    # can't connect to it.
    services.wait_for_redis_to_start(
        redis_ip_address, int(redis_port), password=None)

    # Create a Redis client.
    redis_client = services.create_redis_client(
        redis_address, password=None)

    # Check that the verion information on this node matches the version
    # information that the cluster was started with.
    services.check_version_info(redis_client)

    # Get the node IP address if one is not provided.
    ray_params.update_if_absent(
        node_ip_address=services.get_node_ip_address(redis_address))
    logger.info("Using IP address {} for this node.".format(
        ray_params.node_ip_address))
    # Check that there aren't already Redis clients with the same IP
    # address connected with this Redis instance. This raises an exception
    # if the Redis server already has clients on this node.
    check_no_existing_redis_clients(ray_params.node_ip_address,
                                    redis_client)
    ray_params.update(redis_address=redis_address)
    node = ray.node.Node(ray_params, head=False, shutdown_at_exit=False)
    logger.info("\nStarted Ray on this node. If you wish to terminate the "
                "processes that have been started, run\n\n"
                "    ray stop")

    while True:
        print('.', end='')
        sys.stdout.flush()
        time.sleep(5)
        try:
            redis_client.ping()
        except redis.exceptions.ConnectionError:
            print()
            print("Connection closed.")
            break

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
