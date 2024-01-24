from datasets import Dataset

import utils

assert __name__ == '__main__', 'Cannot be invoked as a module'

dataset = Dataset.from_dict({
    'title': [input('Title=')],
    'original_language': ['en'],  # The training data only consists of English movies
    # 'budget': input("Film's budget (Write 'NA' if not known)="),
    'story': [input('The story/overview in a single line=')],
})

model_output = utils.run_predictions(dataset)
predictions, chances = model_output['prediction'][0], model_output['chance'][0]
# hit_flop_chance_percent = [utils.get_rounded_string(chance.item(), 1) for chance in chances.squeeze()]
print(f'The movie has hit chance of {chances[1]}% and flop chance of {chances[0]}%. '
      f'Therefore it would probably be a {["flop", "hit"][predictions]}.')
