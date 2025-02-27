import pickle
import numpy as np
import pandas as pd

def load_pickle_file(filename):

    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def explore_data(data):
   
    print("\n🔍 Estrutura dos Dados:\n")
    print(f"Total de síndromes: {len(data)}")
    
    for syndrome_id, subjects in list(data.items())[:2]:  # Exibir apenas 2 síndromes
        print(f"Síndrome: {syndrome_id} - Total de pacientes: {len(subjects)}")
        for subject_id, images in list(subjects.items())[:1]:  # Exibir apenas 1 paciente
            print(f"  Paciente: {subject_id} - Total de imagens: {len(images)}")
            for image_id, embedding in list(images.items())[:1]:  # Exibir apenas 1 imagem
                print(f"    Imagem: {image_id} - Dimensão do embedding: {len(embedding)}")
                print(f"    Exemplo de embedding: {embedding[:5]}... (mostrando 5 valores)")
                break
            break
        break

if __name__ == "__main__":
    filename = "data/mini_gm_public_v0.1.p" 
    try:
        data = load_pickle_file(filename)
        explore_data(data)
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
