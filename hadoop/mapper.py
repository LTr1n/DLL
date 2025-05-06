import sys

def main():
    for line in sys.stdin:
        line = line.strip()
        fields = line.split(',')  # Giả sử các cột được phân tách bằng dấu phẩy
        symptoms = fields[2:]  # Giả sử triệu chứng bắt đầu từ cột thứ 3
        for symptom in symptoms:
            print("{}\t1".format(symptom))

if __name__ == "__main__":
    main()