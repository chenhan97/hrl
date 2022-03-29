import json
from random import sample


def compare_list(main_list, sub_list):
    if len(sub_list) == 1:
        if sub_list[0] in main_list:
            return main_list.index(sub_list[0]), main_list.index(sub_list[0])
        else:
            return None, None
    else:
        for i in range(len(main_list)):
            if main_list[i] == sub_list[0]:
                flag = True
                if len(sub_list) + i > len(main_list):
                    flag = False
                else:
                    for j in range(i, len(sub_list) + i):
                        if main_list[j] != sub_list[j - i]:
                            flag = False
                            break
                if flag:
                    return i, len(sub_list) + i - 1
        return None, None


with open("dataset/ter_mul/4/test.json") as infile:
    data = json.load(infile)

ent_list = []
for d_no, d in enumerate(data):
    sent_list = d['token']

    if d['first_start'] == d['first_end']:
        first_entity = [sent_list[d['first_start']]]
    else:
        first_entity = sent_list[d['first_start']:d['first_end'] + 1]

    if d['second_start'] == d['second_end']:
        second_entity = [sent_list[d['second_start']]]
    else:
        second_entity = sent_list[d['second_start']:d['second_end'] + 1]

    if d['third_start'] == d['third_end']:
        third_entity = [sent_list[d['third_start']]]
    else:
        third_entity = sent_list[d['third_start']:d['third_end'] + 1]

    if first_entity not in ent_list:
        ent_list.append(first_entity)
    if second_entity not in ent_list:
        ent_list.append(second_entity)
    if third_entity not in ent_list:
        ent_list.append(third_entity)

with open("dataset/ter_mul/0/new_test.json", mode='w') as outfile:
    out_json_list = []
    count = 0
    for d_no, d in enumerate(data):
        if d_no % 2 == 0:
            tent_ent_list = ent_list.copy()
            out_json_list.append(d)
            sent_list = d['token']
            if d['first_start'] == d['first_end']:
                first_entity = [sent_list[d['first_start']]]
            else:
                first_entity = sent_list[d['first_start']:d['first_end'] + 1]
            if d['second_start'] == d['second_end']:
                second_entity = [sent_list[d['second_start']]]
            else:
                second_entity = sent_list[d['second_start']:d['second_end'] + 1]
            if d['third_start'] == d['third_end']:
                third_entity = [sent_list[d['third_start']]]
            else:
                third_entity = sent_list[d['third_start']:d['third_end'] + 1]
            tent_ent_list.remove(first_entity)
            if first_entity != second_entity:
                tent_ent_list.remove(second_entity)
            if third_entity != second_entity and third_entity != third_entity:
                tent_ent_list.remove(third_entity)
            candi_ent_list = []
            for ent in tent_ent_list:
                start, end = compare_list(sent_list, ent)
                if start is not None and end is not None:
                    candi_ent_list.append([start, end])
            if len(candi_ent_list) < 3:
                count += 1
            else:
                new_entities = sample(candi_ent_list, 3)
                out_json_list.append({"id": d['id'], "token": d['token'], "stanford_pos": d["stanford_pos"],
                                      "stanford_head": d["stanford_head"], "stanford_deprel": d["stanford_deprel"],
                                      "first_start": new_entities[0][0], "first_end": new_entities[0][1],
                                      "second_start": new_entities[1][0], "second_end": new_entities[1][1],
                                      "third_start": new_entities[2][0], "third_end": new_entities[2][1],"relation":"None"})
        else:
            out_json_list.append(d)
    json.dump(out_json_list, outfile)


