import csv

def classifier(row):
	if row[1] <= 7.87:
		if row[3] <= 5.01:
			return '1'
		else:
			if row[1] <= 4.94:
				return '0'
			else:
				return '1'
	else:
		return '0'

def main():
	path = 'C:/abc.csv'
	testing_data = list(csv.reader(open(path)))

	for row in range(1,len(testing_data)):
		for col in range(0, len(testing_data[1])):
			testing_data[row][col] = float(testing_data[row][col])
	testing_data.pop(0)

	with open('C:/xyz.csv', 'w', newline='') as output:
		writer = csv.writer(output)

		for row in testing_data:
			target = classifier(row)
			print(target)
			writer.writerow(target)

main()