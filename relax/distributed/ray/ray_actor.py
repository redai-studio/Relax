from relax.utils.misc import get_current_node_ip, get_free_port


class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1, max_port=None):
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive, max_port=max_port)

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
