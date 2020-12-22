import stomp
import json

conn = stomp.Connection()
conn.connect( wait=True)
data_to_send = {"path": "images/G0138038.JPG",
                "idCaptura": str(1),
                "camara": "CAMARA1"}
conn.send(body=json.dumps(data_to_send), destination='/queue/test')
conn.disconnect()
