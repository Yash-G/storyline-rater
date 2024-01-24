import csv
import datetime
import os
import re
import time
import warnings
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from bs4 import BeautifulSoup, Tag

from config import NA_STRING, MY_LANGUAGES

assert __name__ == '__main__', 'Cannot be invoked as a module'

WIKIPEDIA_HIT_DELAY = 0
CURRENCY_EXCHANGE_RATE = {
    '£': 1.27,
    '$': 1,
    'US$': 1,
    '₹': 0.012,
    '€': 1.09,
    'CAD ': 0.74,
    'NZ$': 0.62,
    'A$': 0.67,
}


def generate_processed_data():
    output_file_path = Path('movies_data.csv')
    is_output_file_fresh = not output_file_path.is_file()
    with (
        open('movies_metadata.csv', newline='', encoding='utf-8') as input_dataset_file,
        open(output_file_path, newline='', mode='a', encoding='utf-8') as output_file,
        open('rows_to_skip.txt') as skip_rows_file
    ):
        file_contents = skip_rows_file.readline().strip()
        rows_to_skip = {int(index_string) for index_string in file_contents.split(',')} if file_contents != '' else set()
        citation_regex = r'(?:\s?\[(?:(?:N )?\d+?|citation needed|inconsistent|unreliable source\?|[a-z])](?:: \d+?)?)*?'
        cited_number_regex = fr'([\d,]+(?:\.\d+)?){citation_regex}'
        multiplier_dict = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9}
        multiplier_regex = '|'.join(multiplier_dict.keys())
        currency_regex = '|'.join([re.escape(key) for key in CURRENCY_EXCHANGE_RATE.keys()])
        amount_pattern = re.compile(rf'(?:(?:Domestic|about |Less than |>)?({currency_regex}){cited_number_regex}(?:[-–]{cited_number_regex})?'
                                    rf'(?:\s({multiplier_regex}))?(?: gross)?(?:\s?\(.+?\))?{citation_regex}(?:\(.+?\))?)+?')

        def parse_amount_string(tag_to_parse: Tag) -> int:
            amount_string = tag_to_parse.next_sibling.text.strip().split('\n')[-1]
            result = amount_pattern.fullmatch(amount_string)
            assert result is not None, (f'amount string {amount_string}, does not match'
                                        f' the amount pattern {amount_pattern}')
            currency, lhs_multiplier_string, rhs_multiplier_string, multiplicand_string = result.groups()
            lhs_multiplier, rhs_multiplier = [
                float(multiplier_string.replace(',', '_')) if multiplier_string is not None else None
                for multiplier_string in [lhs_multiplier_string, rhs_multiplier_string]
            ]
            multiplicand = multiplier_dict[multiplicand_string] if multiplicand_string is not None else 1
            assert multiplicand is not None, (f'Unknown multiplicand identifier {multiplicand_string} is present'
                                              f'in the amount string {amount_string}')
            if rhs_multiplier is None:
                multiplier = lhs_multiplier
            else:
                multiplier = (lhs_multiplier + rhs_multiplier) / 2
            return round(multiplier * multiplicand * CURRENCY_EXCHANGE_RATE[currency])

        previous_failed_indexes: set[int]
        with open('previous_run.txt') as previous_run_file:
            processed_row_count = int(previous_run_file.readline().strip())
            line_2 = previous_run_file.readline().strip()
            previous_failed_indexes = {int(index_text) for index_text in line_2.split(',')} if line_2 != '' else set()
        output_writer: csv.DictWriter[str] | None = None
        current_failed_indexes: set[int] = set()
        row_index = -1
        row: dict[str, str]
        try:
            for row in csv.DictReader(input_dataset_file):
                row_index += 1
                if (row_index < processed_row_count and row_index not in previous_failed_indexes) or row_index in rows_to_skip:
                    print(f'Skipping {row}')
                    continue
                title = row['title']
                original_language = row['original_language']
                def map_zero_to_none(column: str) -> int:
                    value = int(row[column])
                    return value if value != 0 else None
                budget, revenue = map_zero_to_none('budget'), map_zero_to_none('revenue')
                story = row['overview']
                is_story_wiki_plot = False
                if original_language not in MY_LANGUAGES:
                    print(f'Skipping movie {title} with original language {original_language}')
                    continue
                given_wikipedia_id = row['wikipedia_id']
                if given_wikipedia_id != '':
                    wikipedia_ids = [given_wikipedia_id]
                else:
                    underscored_title = title.replace(' ', '_')
                    quoted_title = quote(underscored_title)
                    wikipedia_ids = [f'{quoted_title}_(film)', quoted_title]
                    release_date_string = row['release_date']
                    if release_date_string != '':
                        release_year = datetime.datetime.strptime(release_date_string, '%d-%m-%Y').year
                        wikipedia_ids.insert(0, f'{quoted_title}_({release_year}_film)')
                for wiki_id_index, wiki_id in enumerate(wikipedia_ids):
                    url = f'https://en.wikipedia.org/wiki/{wiki_id}'
                    is_wiki_id_last = wiki_id_index == len(wikipedia_ids) - 1
                    try:
                        with urlopen(url) as response:
                            time.sleep(WIKIPEDIA_HIT_DELAY)
                            soup = BeautifulSoup(response, 'html.parser')
                            tag_candidates = soup.find_all(id=['Plot', 'Characters'])
                            match len(tag_candidates):
                                case 0:
                                    if not is_wiki_id_last:
                                        continue
                                    else:
                                        error_message = (f'Unable to locate story for "{title}". '
                                                         f'' f'Tried with wikipedia ids {wikipedia_ids}')
                                        warnings.warn(error_message)
                                        # current_failed_indexes.add(row_index)
                                        plot_inner_tag = None
                                case 1:
                                    plot_inner_tag = tag_candidates[0]
                                case _:
                                    assert len(tag_candidates) == 2
                                    if tag_candidates[0]['id'] == 'Plot':
                                        plot_inner_tag = tag_candidates[0]
                                    else:
                                        plot_inner_tag = tag_candidates[1]
                            if plot_inner_tag is not None:
                                plot_header_tag = plot_inner_tag.parent
                                tag: Tag
                                plot_tags = []
                                for tag in plot_header_tag.next_siblings:
                                    if tag == '\n' or tag.text == '' or tag.name in ['figure', 'style', 'table'] or (
                                            tag.name == 'div' and ('thumb' in tag.get('class') or 'note' == tag.get('role'))):
                                        continue
                                    elif tag.name in ['p', 'ul', 'ol', 'blockquote', 'dl'] or (tag.name == 'div' and 'quotebox' in tag.get('class')):
                                        plot_tags.append(tag.text)
                                    elif tag.name.startswith('h'):
                                        tag_suffix = tag.name[1:]
                                        if tag_suffix.isdigit() and tag_suffix > '2':
                                            plot_tags.append(tag.text)
                                        else:
                                            break
                                    else:
                                        assert False, (f'Plot h2 header not followed by another p, h2, h3, h4... tag,'
                                                       f' it is followed by {tag}')
                                plot = '\n'.join(plot_tags)
                                if plot == '':
                                    warnings.warn(f'Plot is empty at {url}. Please check once.')
                                else:
                                    story = plot
                                    is_story_wiki_plot = True
                            wiki_budget, wiki_revenue = None, None
                            for tag in soup.find_all(attrs={'class', 'infobox-label'}):
                                match tag.text:
                                    case 'Budget':
                                        assert wiki_budget is None
                                        wiki_budget = parse_amount_string(tag)
                                    case 'Box office':
                                        assert wiki_revenue is None
                                        wiki_revenue = parse_amount_string(tag)
                            if wiki_revenue is not None:
                                revenue = wiki_revenue
                            if wiki_budget is not None:
                                budget = wiki_budget

                    except HTTPError as error:
                        assert error.code == 404
                        if is_wiki_id_last:
                            warnings.warn((f'Unable to locate Wikipedia URL for "{title}". '
                                       f'' f'Tried with wikipedia ids {wikipedia_ids}'))
                            # current_failed_indexes.add(row_index)
                        else:
                            continue

                    if revenue is not None and budget is not None:
                        profit = revenue - budget
                        is_good = profit / budget > 0.25
                    else:
                        profit = 'NA'
                        is_good = False

                    output_row = {
                        'id': row['id'],
                        'original_language': original_language,
                        'title': title,
                        'story': story,
                        'is_story_detailed': is_story_wiki_plot,
                        'budget': budget if budget is not None else NA_STRING,
                        'revenue': revenue if revenue is not None else NA_STRING,
                        'profit': profit if profit is not None else NA_STRING,
                        'is_good': is_good,
                        'url': url,
                    }
                    if output_writer is None:
                        output_writer = csv.DictWriter(output_file, output_row.keys())
                        if is_output_file_fresh:
                            output_writer.writeheader()
                            if processed_row_count > 0:
                                warnings.warn(f'Output file is newly created, however, {processed_row_count}'
                                              f'records have already been processed')
                    output_writer.writerow(output_row)
                    print('Wrote', output_row)
                    break

            row_index += 1
        finally:
            if row_index > processed_row_count or previous_failed_indexes != current_failed_indexes:
                line_1 = str(row_index) + os.linesep
                line_2 = ','.join([str(index) for index in current_failed_indexes]) + os.linesep
                with open('previous_run.txt', mode='w') as previous_run_file:
                    previous_run_file.writelines([line_1, line_2])


generate_processed_data()
