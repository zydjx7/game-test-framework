from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
import cv2
import LogicLayer
import os
import sys

# Initialize the Flask application
app = Flask(__name__)

# 从环境变量获取测试目标，默认为 'assaultcube'
target_name = os.getenv("GAMECHECK_TARGET", "assaultcube")
print(f"[Flask] 使用测试目标: {target_name}")

# 初始化 LogicLayer 实例，传递测试目标
logicLayer_inst = LogicLayer.LogicLayer(target_name=target_name)

@app.route('/index', methods=['GET', 'POST'])
def index():
    return jsonify(200)

    """ # Transforms to a dictionary 
    return jsonify(username=g.user.username,
                   id=g.user.id)
    """

# route http posts to this method
@app.route('/check_visuals', methods=['GET', 'POST'])
def check_visuals():
    # Get the message
    r = request

    msg_received = jsonpickle.decode(r.data)
    # print(msg_received)
    assert (isinstance(msg_received, dict))

    print("Received keys ", msg_received.keys())

    testContext = msg_received["testContext"]
    requestCategory = testContext["requestCategory"]
    requestFunc = testContext["requestFunc"]
    requestExpectedAnswer = msg_received["expectedAnswer"]

    if requestCategory == "UI":
        res = testUI(context = testContext,
                     requestFunc = requestFunc,
                     requestExpectedAnswer = requestExpectedAnswer)

        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(res)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    else:
        raise NotImplementedError("Not implemented")

# Test the results
def testUI(context, requestFunc, requestExpectedAnswer):
    # Step 1: decode the screenshots from context
    #--------------------------------------------------
    numScreenshots = context["screenshotsCount"]
    screenshotData = context["screenshotData"]
    # TODO: fix multiple screenshots sending
    
    try:
        # 将hex字符串转换回bytes
        bytes_data = bytes.fromhex(screenshotData)
        # 然后再转换为numpy数组
        img_asnumpy = np.frombuffer(bytes_data, np.uint8)
        
        # decode image
        imgs = [cv2.imdecode(img_asnumpy, cv2.IMREAD_COLOR)]
        
        if imgs[0] is None:
            print("警告: 图像解码失败，请检查数据格式")
            return {'result': False, 'message': '图像解码失败'}
    except Exception as e:
        print(f"图像解码错误: {str(e)}")
        return {'result': False, 'message': f'图像解码错误: {str(e)}'}

    # Step 2: check the type of test and conduct the appropiate check, get results
    res = 0
    if requestFunc == "checkWeaponCross":
        res = logicLayer_inst.testWeaponCrossPresence(imgs, context, requestExpectedAnswer)
    elif requestFunc == "checkAmmoSyncText":
        res = logicLayer_inst.testAmmoTextInSync(imgs, context, requestExpectedAnswer)
    else:
        raise NotImplementedError("Not implemented")

    # Pack the result and respond
    response = {'result': res}
    return response


# TODO add all other functionalities
@app.route('/check_sounds', methods=['GET', 'POST'])
def check_sounds():
    # Pack the result and respond
    response = {'result': 0} 
    return response

# 只在直接运行时启动Flask服务器，作为模块导入时不启动
if __name__ == '__main__':
    try:
        # 读取环境变量可选指定端口，默认5000
        port = int(os.getenv("GAMECHECK_PORT", "5000"))
        print(f"[Flask] starting on port {port}")
        
        # 禁用dotenv自动加载，避免编码问题
        os.environ["FLASK_SKIP_DOTENV"] = "1"
        
        # 启动Flask应用
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Flask服务器启动失败: {e}", file=sys.stderr)
        sys.exit(1)

