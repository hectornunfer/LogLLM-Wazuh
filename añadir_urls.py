import pandas as pd
import ast
import urllib.parse
from decimal import Decimal


# Ruta del archivo CSV
csv_file = "detections.csv"

# Cargar el archivo CSV
try:
    df = pd.read_csv(csv_file)
    
    # Función para generar la URL de búsqueda en Wazuh
    def generate_wazuh_url(wazuh_id_str):
        # Convertir el string de lista en una lista real        
        # Construcción de la cadena de parámetros
        filter_values = wazuh_id_str.replace("[","").replace("]","").replace(",","','").replace('"',"'")
        query_values = wazuh_id_str.replace("[","").replace("]","").replace(",",",%20")
        lista_ids = [Decimal(num) for num in wazuh_id_str.replace("[","").replace("]","").split(",")]

        # Construcción de la parte 'should' separada
        should_part = ",".join(f"(match_phrase:(id:'{v}'))" for v in lista_ids)

        # Construcción del filtro principal
        filter_param = (
            f"filters:!(('$state':(store:appState),"
            f"meta:(alias:!n,disabled:!f,index:'wazuh-alerts-*',key:id,negate:!f,"
            f"params:!('{filter_values}'),type:phrases,value:'{query_values}'),"
            f"query:(bool:(minimum_should_match:1,should:!({should_part})))))"
            f",query:(language:kuery,query:''))&_g=(filters:!(),refreshInterval:(pause:!t,value:0),"
            f"time:(from:now-1M,to:now))"
        )                
        # Construcción de la URL final
        base_url = "https://34.175.254.53/app/threat-hunting#/overview/?tab=general&tabView=events&_a=("
        return base_url + filter_param
    
    # Aplicar la función a cada fila y crear una nueva columna con la URL
    df["Wazuh_URL"] = df["Wazuh_ID"].apply(generate_wazuh_url)
    
    # Guardar el archivo con la nueva columna
    df.to_csv(csv_file, index=False)
    print("Archivo actualizado con las URLs.")
except Exception as e:
    print(f"Error: {e}")
