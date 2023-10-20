from datasets import load_dataset

dataset = load_dataset("alkzar90/CC6204-Hackaton-Cub-Dataset")
split = 'test'
iterable_dataset = load_dataset("alkzar90/CC6204-Hackaton-Cub-Dataset", split=split)

num_of_samples = 0
list_of_ids, list_of_filenames = [], []
list_of_labels = []
list_of_images = []
list_of_textprompts = []


for sample in iterable_dataset:
    if num_of_samples > 1000:
        break
    if not sample['description'] or not sample['file_name'] or sample['description'] == '' or sample['file_name'] == '':
        continue
    
    list_of_ids.append(num_of_samples)
    list_of_labels.append(sample['label'])
    list_of_filenames.append(sample['file_name'])
    list_of_images.append(sample['image'])

    sentences = sample['description'].replace('\n', ' ').split(". ")
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    output_string = ". ".join(capitalized_sentences)
    list_of_textprompts.append("A picture of a bird. " + output_string)
    num_of_samples += 1
print(list_of_labels[0:10])

print(list_of_filenames[0:10])
