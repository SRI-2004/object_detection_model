import csv

# Specify the range of numbers from 1 to 1500
numbers = list(range(1, 1001))

# Open a CSV file in write mode
with open('traink.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)

    # Write header row
    csvwriter.writerow(['Image', 'Text'])

    # Write data rows
    for number in numbers:
        # Format the numbers with leading zeros
        image_filename = f"{number:06d}.jpg"
        text_filename = f"{number:06d}.txt"

        # Write the row to the CSV file
        csvwriter.writerow([image_filename, text_filename])

print("CSV file 'output.csv' created successfully.")
