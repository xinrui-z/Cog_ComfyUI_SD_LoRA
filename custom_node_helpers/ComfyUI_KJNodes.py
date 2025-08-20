from custom_node_helper import CustomNodeHelper

class ComfyUI_KJNodes(CustomNodeHelper):
    @staticmethod
    def add_weights(weights_to_download, node):
        if node.is_type_in(["BatchCLIPSeg", "DownloadAndLoadCLIPSeg"]):
            weights_to_download.extend(["models--CIDAS--clipseg-rd64-refined"])
