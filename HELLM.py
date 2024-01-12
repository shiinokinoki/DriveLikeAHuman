import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from LLMDriver.customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe,
)

# memo: apiがazureかopenaiか, apiのkeyなどをconfig.yamlから読み込む
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

# memo: azureの時は今回使用しないので無視 
if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
    os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
    os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['AZURE_API_VERSION']
    os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['AZURE_API_BASE']
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['AZURE_API_KEY']
    llm = AzureChatOpenAI(
        deployment_name=OPENAI_CONFIG['AZURE_MODEL'],
        temperature=0,
        max_tokens=1024,
        request_timeout=60
    )
# memo: openaiの時はkeyを設定
elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    # 環境変数にkeyを設定
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
    # モデルを設定 (chatモデルを使用)
    # LangChainにはLLMとChatModelが存在。
    # LLMは文字列入力に対して文字列を返すが、ChatModelは"メッセージ"のリストを入力として受け取り、"メッセージ"を出力として返す.
    # "メッセージ"はContentとRoleを含む。
    # Roleには、AI, System, Human, (Function, Tool)がある。
    # max tokensは、出力の最大文字数。
    # temperatureは、出力の多様性を調整するパラメータ。0で固定化？される。
    # modelは8000以上のcontext入力が許可されているモデルを使用する必要がある。
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-1106', # or any other model with 8k+ context
        max_tokens=1024,
        request_timeout=60
    )


# base setting
vehicleCount = 15

# Highway-v0（環境システム）二巻する設定
# environment setting
# ACTIONS_ALL = {
#         0: 'LANE_LEFT',
#         1: 'IDLE',
#         2: 'LANE_RIGHT',
#         3: 'FASTER',
#         4: 'SLOWER'
#     }
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "vehicles_count": vehicleCount,
        "see_behind": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration": 40,
    "vehicles_density": 2,
    "show_trajectories": True,
    "render_agent": True,
}

# 高速道路環境の定義
env = gym.make('highway-v0', render_mode="rgb_array")
env.configure(config)
# 環境の録画
env = RecordVideo(
    env, './results-video',
    name_prefix=f"highwayv0"
)
env.unwrapped.set_record_video_wrapper(env)
# 初期環境の観察
obs, info = env.reset()
# 実行中に環境の状態をwindowで表示する
env.render()

# scenario and driver agent setting
if not os.path.exists('results-db/'):
    os.mkdir('results-db')
database = f"results-db/highwayv0.db"
sce = Scenario(vehicleCount, database)
toolModels = [
    getAvailableActions(env),
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
    isActionSafe(),
]
DA = DriverAgent(llm, toolModels, sce, verbose=True)
outputParser = OutputParser(sce, llm)
output = None
done = truncated = False
frame = 0
try:
    while not (done or truncated):
        sce.upateVehicles(obs, frame)
        DA.agentRun(output)
        da_output = DA.exportThoughts()
        output = outputParser.agentRun(da_output)
        env.render()
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()
        obs, reward, done, info, _ = env.step(output["action_id"])
        print(output)
        frame += 1
finally:
    env.close()