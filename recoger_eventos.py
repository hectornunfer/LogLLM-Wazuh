import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
import json

USER = 'admin'  # Reemplaza con tu usuario
PASSWORD = 'Ps5.CG*ZWnN64BuDme+vk6fLFJ4ggYYF'  # Reemplaza con tu contraseña
WAZUH_INDEXER_IP = '34.175.254.53'  # Reemplaza con la IP de tu Wazuh Indexer
URL = f"https://{WAZUH_INDEXER_IP}:9200/_search?scroll=1m"
AGENT_NAME = "agente-thunderbird"  # Nombre del agente a consultar

# Calcular el rango de tiempo (últimas 24 horas)
now = datetime.utcnow()
one_day_ago = now - timedelta(minutes=2)

# Formatear fechas en el formato esperado por Elasticsearch
time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
query_time_range = {
    "gte": one_day_ago.strftime(time_format),
    "lte": now.strftime(time_format)
}

# Cuerpo de la consulta
query_body = {
    "size": 10000,  # Ajusta según la cantidad esperada
    "_source": ["id", "full_log", "timestamp"],  # Solo recuperar estos campos
    "query": {
        "bool": {
            "must": [
                {"term": {"agent.name": AGENT_NAME}},  # Filtrar por nombre del agente
                {"term": {"rule.description": "Thunderbird log."}},
                {"range": {"@timestamp": query_time_range}}  # Filtrar por rango de tiempo
            ]
        }
    }
}

# Realizar la primera solicitud con scroll
response = requests.post(
    URL,
    auth=HTTPBasicAuth(USER, PASSWORD),
    headers={"Content-Type": "application/json"},
    data=json.dumps(query_body),
    verify=False
)

if response.status_code == 200:
    results = response.json()
    scroll_id = results["_scroll_id"]
    hits = results["hits"]["hits"]

    # Extraer solo los campos requeridos
    alertas = [
        {
            "id": alerta["_source"].get("id", "N/A"),
            "full_log": alerta["_source"].get("full_log", "N/A"),
            "timestamp": alerta["_source"].get("timestamp", "N/A")
        }
        for alerta in hits
    ]

    # Guardar los full_log en Thunderbird.log
    with open("Thunderbird.log", "w", encoding="utf-8") as log_file:
        for alerta in alertas:
            log_file.write(alerta["id"] + " " + alerta["full_log"] + "\n")

    # Seguir obteniendo datos mientras haya más resultados
    while len(results["hits"]["hits"]) > 0:
        scroll_response = requests.post(
            f"https://{WAZUH_INDEXER_IP}:9200/_search/scroll",
            auth=HTTPBasicAuth(USER, PASSWORD),
            headers={"Content-Type": "application/json"},
            data=json.dumps({"scroll": "1m", "scroll_id": scroll_id}),
            verify=False
        )

        if scroll_response.status_code == 200:
            results = scroll_response.json()
            new_hits = results["hits"]["hits"]
            new_alertas = [
                {
                    "id": alerta["_source"].get("id", "N/A"),
                    "full_log": alerta["_source"].get("full_log", "N/A"),
                    "timestamp": alerta["_source"].get("timestamp", "N/A")
                }
                for alerta in new_hits
            ]
            alertas.extend(new_alertas)

            # Guardar los nuevos full_log en Thunderbird.log
            with open("Thunderbird.log", "a", encoding="utf-8") as log_file:
                for alerta in new_alertas:
                    log_file.write(alerta["id"] + " " + alerta["full_log"] + "\n")
        else:
            break

    # Guardar las alertas en un archivo JSON
    with open("alertas.json", "w", encoding="utf-8") as f:
        json.dump(alertas, f, indent=2, ensure_ascii=False)

    print(f"Total alertas obtenidas: {len(alertas)}")
    print("Alertas guardadas en alertas.json")
    print("Logs guardados en Thunderbird.log")
else:
    print(f"Error: {response.status_code}")
    print(response.text)