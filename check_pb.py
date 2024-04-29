import tensorflow as tf

# 그래프 파일 로드
with tf.io.gfile.GFile('./frozen_model/ocr_graph_optimized.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# 그래프 내용 출력
# for node in graph_def.node:
    # print(node.name)  # 노드 이름 출력

# 특정 노드의 세부 정보를 출력하려면
print([node for node in graph_def.node if node.name == 'map/while/Switch'])
