import time
import os

input_file = "Thunderbird.log"
output_file = "/var/log/thunder/Thunderbird.log"

def main():
    if not os.path.exists(input_file):
        print(f"El archivo de entrada '{input_file}' no existe.")
        return

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_file, "r") as infile, open(output_file, "a") as outfile:
            for line in infile:
                outfile.write(line)
                outfile.flush()
                print(f"Línea escrita: {line.strip()}")
                time.sleep(0.3)

        print("Procesamiento completado.")
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")

if __name__ == "__main__":
    main()