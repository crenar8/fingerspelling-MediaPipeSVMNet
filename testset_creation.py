import os
import shutil

destination_root = "image_dataset/test"
source_root = "image_dataset/training"

# Sub-folders retrieving
subfolders = os.listdir(source_root)

for subfolder in subfolders:
    source_folder_path = os.path.join(source_root, subfolder)

    if os.path.isdir(source_folder_path):
        destination_folder_path = os.path.join(destination_root, subfolder)

        # Crea la sottocartella di destinazione se non esiste
        os.makedirs(destination_folder_path, exist_ok=True)

        # Ottieni la lista dei file nella sottocartella corrente
        files = sorted(os.listdir(source_folder_path))

        # Seleziona le prime 20 immagini ogni 100
        selected_files = [files[i] for i in range(len(files)) if i % 100 < 20]

        # Sposta i file selezionati nella cartella di destinazione corrispondente
        for file_name in selected_files:
            source_file_path = os.path.join(source_folder_path, file_name)
            destination_file_path = os.path.join(destination_folder_path, file_name)
            shutil.move(source_file_path, destination_file_path)
