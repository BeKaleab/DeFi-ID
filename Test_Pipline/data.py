import csv

def merge(input, output):
    with open(input, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output, 'a') as new_file:
            fieldnames = csv_reader.fieldnames
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            for line in csv_reader:
                csv_writer.writerow(line)

def copy(input, output):
    with open(input, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output, 'w') as new_file:
            fieldnames = csv_reader.fieldnames
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for line in csv_reader:
                csv_writer.writerow(line)

def remove(input, output, del_field):
    with open(input, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output, 'w') as new_file:
            fieldnames = list(set(csv_reader.fieldnames) - set(del_field))
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for line in csv_reader:
                for f in del_field:
                    del line[f]
                csv_writer.writerow(line)

def add(input, output, del_field, field_val):
    with open(input, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output, 'w') as new_file:
            fieldnames = csv_reader.fieldnames + del_field
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for line in csv_reader:
                for i, f in enumerate(del_field):
                    line[f] = field_val[i][csv_reader.line_num-2]
                csv_writer.writerow(line)

def add_empty(input, output, del_field):
    with open(input, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output, 'w') as new_file:
            fieldnames = csv_reader.fieldnames + del_field
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for line in csv_reader:
                for i, f in enumerate(del_field):
                    line[f] = ''
                csv_writer.writerow(line)

def file_size(input):
    with open(input, 'r') as csv_file:
        return len(csv_file.readlines())

def rename(input, output, oldfn, newfn):
    with open(input, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output, 'w') as new_file:
            fieldnames = []
            for f in csv_reader.fieldnames:
                fn = f
                for i, n in enumerate(oldfn):
                    if(fn == n):
                        fn = newfn[i]
                fieldnames.append(fn)
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for line in csv_reader:
                for i, f in enumerate(csv_reader.fieldnames):
                    if f != csv_writer.fieldnames[i]:
                        line[csv_writer.fieldnames[i]] = line.pop(f)
                csv_writer.writerow(line)

def update(input, output, fid, old_name_new_data, fn='email'):
    with open(input, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output, 'w') as new_file:
            fieldnames = csv_reader.fieldnames
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for line in csv_reader:
                for i, v in enumerate(fid):
                    if line[fn] == v:
                        for updates in old_name_new_data[i]:
                            line[updates[0]] = updates[1]
                csv_writer.writerow(line)

#merge('input.csv','output.csv')
#copy('input.csv','output.csv')
#remove('input.csv','output.csv',['last_name','email'])
fz = file_size('input.csv')-1
print(fz)
add_empty('input.csv','output.csv',['address','income'])
add('input.csv','output.csv',['address','income'],[['UCB']*fz,['$0']*fz])
update('input.csv','output.csv',['nicolejacobs@bogusemail.com','maggiepatterson@bogusemail.com'],[[('first_name','N1'),('last_name','A1')],[('first_name','N2'),('last_name','A2')]] )
#rename('input.csv','output.csv',['first_name','last_name'],['fn','ln'])