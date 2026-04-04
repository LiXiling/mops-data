from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda import PandaWristCam


@register_agent()
class PandaLongHand(PandaWristCam):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_longhand"
    urdf_path = "/home/max/code/affordance-gen/src/mops/resources/panda_v3.urdf"
