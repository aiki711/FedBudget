# src/mlflow_config.py
import mlflow

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")    # ← 1 と同じホスト名/ポート
    mlflow.set_experiment("prophet_baseline")           # スクリプトごとに実験名を変える

#自分の PC だけで使う	--host localhost で OK。外部ブラウザからはアクセス不可。
#チーム LAN 内で共有	--host 0.0.0.0 + 会社 LAN の IP でアクセス。
#                      必要なら Windows Defender / ufw で TCP 5000 を許可。
#クラウド (EC2/GCE)	    同上 + セキュリティグループでポート 5000 を開放。
#                      SSL 終端なら Nginx+Let'sEncrypt を前段に置く。