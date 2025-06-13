import os
import s3fs

fs = s3fs.S3FileSystem(
client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
key = os.environ["AWS_ACCESS_KEY_ID"], 
secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
token = os.environ["AWS_SESSION_TOKEN"])


def push_files_to_s3_with_nested_folders(bucket_name="debie", local_dir=None, s3_base_folder="", file_extensions=None, overwrite=False, fs=None):
    if file_extensions is None:
        file_extensions = ['csv', 'pdf']

    # Find all files with the specified extensions in local_dir and its subdirectories
    files_to_upload = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            file_extension = file.split('.')[-1].lower()
            if file_extension in file_extensions:
                full_path = os.path.join(root, file)
                files_to_upload.append(full_path)

    # Upload each file to S3
    for file_path in files_to_upload:
        # Calculate the relative path from local_dir
        relative_path = os.path.relpath(file_path, local_dir)

        # Replace backslashes with forward slashes (for Windows compatibility)
        relative_path = relative_path.replace('\\', '/')

        # Construct the S3 key
        s3_key = os.path.join(s3_base_folder, relative_path).replace('\\', '/')

        # Construct the full S3 path
        s3_path = f"{bucket_name}/{s3_key}"

        # Check if the file already exists in S3
        exists = fs.exists(s3_path)

        if exists and not overwrite:
            print(f"File {s3_key} already exists in S3. Skipping upload.")
        else:
            # Upload the file
            fs.put(file_path, s3_path)
            print(f"Uploaded {file_path} to {s3_key}")

    print("End of the push")

# Assuming fs is already set up as in your code

local_dir = "/home/onyxia/work/EstITE/Simulations_Stage/Setup 6/Results/"
s3_base_folder = "Simulations_Stage/Setup 6/Results/"
bucket_name = "debie"  # or the actual bucket name

# Call the function with your fs object
push_files_to_s3_with_nested_folders(
    bucket_name=bucket_name,
    local_dir=local_dir,
    s3_base_folder=s3_base_folder,
    file_extensions=["csv", "pdf"],
    overwrite=False,
    fs=fs
)