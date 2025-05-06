import sys

def main():
    symptom_count = {}
    
    for line in sys.stdin:
        line = line.strip()
        symptom, count = line.split('\t')
        count = int(count)
        
        if symptom in symptom_count:
            symptom_count[symptom] += count
        else:
            symptom_count[symptom] = count
    
    for symptom, total in symptom_count.items():
        print("{}\t{}".format(symptom, total))

if __name__ == "__main__":
    main()
