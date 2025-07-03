if (!require("paws")) {
  install.packages("paws", repos = "https://cloud.R-project.org")
}

Sys.setenv("AWS_ACCESS_KEY_ID" = "secret",
           "AWS_SECRET_ACCESS_KEY" = "secret",
           "AWS_DEFAULT_REGION" = "secret",
           "AWS_SESSION_TOKEN" = "secret",
           "AWS_S3_ENDPOINT"= "secret")

library("paws")
minio <- paws::s3(config = list(
	credentials = list(
	  creds = list(
		access_key_id = Sys.getenv("AWS_ACCESS_KEY_ID"),
		secret_access_key = Sys.getenv("AWS_SECRET_ACCESS_KEY"),
		session_token = Sys.getenv("AWS_SESSION_TOKEN")
	  )),
	endpoint = paste0("https://", Sys.getenv("AWS_S3_ENDPOINT")),
	region = Sys.getenv("AWS_DEFAULT_REGION")))
  
# minio$list_buckets()


###############################################
### Playground to psuh and pull with the S3 ###
###############################################

# Create a sample data frame
data <- data.frame(
  id = 1:10,
  value = rnorm(10)
)

print("orig data")
print(data)

# Write the data frame to a CSV file
write.csv(data, "sample_data.csv", row.names = FALSE)

# Specify the bucket name and object key
bucket_name <- "debie"  # Replace with your bucket name
object_key <- "sample_data_3.csv" # name of the object in the S3

path_to_file <- "folder1/folder2/folder3/"

full_path <- paste0(path_to_file, object_key)

# Upload the file to S3
minio$put_object(Bucket = bucket_name, Key = full_path, Body = "sample_data.csv")


# List objects in the bucket
objects <- minio$list_objects_v2(Bucket = bucket_name)

# Print the list of objects
print(objects$Contents)

# Download the file from S3
downloaded_file <- minio$get_object(Bucket = bucket_name, Key = "sample_data_2.csv")

# Write the downloaded content to a local file
writeBin(downloaded_file$Body, "downloaded_sample_data_2.csv")

# Read the downloaded CSV file
downloaded_data <- read.csv("downloaded_sample_data_2.csv")

# Print the downloaded data
print("dl data")
print(downloaded_data)

##########################################
### Push and pull files to and from S3 ###
##########################################


push_files_to_S3_with_nested_folders <- function(bucket_name = "debie",
					local_dir, s3_base_folder, file_extensions = c("csv", "pdf"),
					overwrite = FALSE, public_read = FALSE){
  # Recursive function to find all files with specified extensions in nested directories
  find_files <- function(dir) {
    # Create a pattern to match any of the specified file extensions
    pattern <- paste0("\\.(", paste(file_extensions, collapse = "|"), ")$")

    files <- list.files(path = dir, pattern = pattern, full.names = TRUE)
    subdirs <- list.dirs(path = dir, full.names = TRUE, recursive = FALSE)

    for (subdir in subdirs) {
      files <- c(files, find_files(subdir))
    }

    files
  }

  # Get all files with specified extensions in the local directory and its subdirectories
  files <- find_files(local_dir)

  # Upload each file to S3, preserving the folder structure
  for (file in files) {
    # Get the relative path of the file with respect to the local directory
    relative_path <- substring(file, nchar(local_dir) + 1)

    # Create the full S3 key (s3_base_folder + relative path)
    s3_key <- paste0(s3_base_folder, relative_path)

    # Ensure the S3 key uses forward slashes (even on Windows)
    s3_key <- gsub("\\\\", "/", s3_key)

    # Check if the file already exists in S3
    exists <- tryCatch({
      minio$head_object(Bucket = bucket_name, Key = s3_key)
      TRUE
    }, error = function(e) {
      FALSE
    })

    if (exists && !overwrite) {
      message(paste("File", s3_key, "already exists in S3. Skipping upload."))
    } else {
      # Set the ACL based on the public_read parameter
      acl <- if (public_read) "public-read" else "private"

      # Upload the file to S3 with the specified ACL
      minio$put_object(Bucket = bucket_name, Key = s3_key, Body = file, ACL = acl)

      message(paste("Uploaded", file, "to", s3_key, "with ACL:", acl))
    }
  }

  print("End of the push")
}


# Specify the local directory containing the CSV files and nested folders
local_dir <- "/home/onyxia/work/EstITE/Simulations_Stage/"

# Specify the base S3 "folder" path
s3_base_folder <- "Simulations_Stage/"

# Push CSV files to S3, preserving the folder structure
push_files_to_S3_with_nested_folders(local_dir = local_dir, s3_base_folder = s3_base_folder,
									file_extensions = c("csv", "pdf"),
									overwrite = FALSE, public_read = TRUE)


pull_files_from_S3_with_nested_folders <- function(bucket_name = "debie",
								s3_base_folder, local_dir, file_extensions = c("csv", "pdf"),
								overwrite = FALSE) {
  # Ensure the local directory exists
  if (!dir.exists(local_dir)) {
    dir.create(local_dir, recursive = TRUE)
  }

  # List objects in the S3 bucket with the specified prefix
  objects <- minio$list_objects_v2(Bucket = bucket_name, Prefix = s3_base_folder)

  # Check if there are any objects
  if (is.null(objects$Contents) || length(objects$Contents) == 0) {
    message("No objects found in the specified S3 folder.")
    return()
  }

  # Create a pattern to match any of the specified file extensions
  pattern <- paste0("\\.(", paste(file_extensions, collapse = "|"), ")$")

  # Download each file with specified extensions, preserving the folder structure
  for (object in objects$Contents) {
    # Check if the file has one of the specified extensions
    if (grepl(pattern, object$Key, ignore.case = TRUE)) {
      # Extract the relative path from the S3 key
      relative_path <- substring(object$Key, nchar(s3_base_folder) + 1)

      # Create the local file path
      local_file <- file.path(local_dir, relative_path)

      # Ensure the local directory exists
      dir.create(dirname(local_file), recursive = TRUE)

      # Check if the local file already exists
      if (file.exists(local_file) && !overwrite) {
        message(paste("File", local_file, "already exists locally. Skipping download."))
      } else {
        # Download the file from S3
        downloaded_file <- minio$get_object(Bucket = bucket_name, Key = object$Key)

        # Write the downloaded content to a local file
        writeBin(downloaded_file$Body, local_file)

        message(paste("Downloaded", object$Key, "to", local_file))
      }
    }
  }

  print("End of the pull")
}



# Specify the local directory containing the CSV files and nested folders
local_dir <- "/home/onyxia/work/EstITE/Simulations_Stage_bis/"

# Specify the base S3 "folder" path
s3_base_folder <- "Simulations_Stage/"

pull_files_from_S3_with_nested_folders(local_dir = local_dir, s3_base_folder = s3_base_folder,
										file_extensions = c("csv", "pdf"), overwrite = FALSE)