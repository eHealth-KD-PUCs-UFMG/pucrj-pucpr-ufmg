# Output
Cada pasta, nomeada com o horário da execução, contém os seguintes arquivos:
- **data.txt**: sentenças utilizadas no teste.
- **evaluation.txt**: avaliação do modelo para todos os cenários.
- **model_parameters.txt**: parâmetros do modelo utilizado.
- **raw_verbose_entity.txt**: saída do teste de identificação de entidades com os casos errados.
- **raw_verbose_relation.txt**: saída do teste de identificação de relações com os casos errados.
- **verbose_entity.xlsx**: tabela com o conteúdo formatado da saída "raw_verbose_entity". Cada aba representa um tipo de erro.
- **verbose_relation.xlsx**: tabela com o conteúdo formatado da saída "raw_verbose_relation". Cada aba representa um tipo de erro.

Abaixo está a explicação do desafio para cada tipo de erro:
### Entity

- **Correct matches** are reported when a text in the dev file matches exactly with a corresponding text span in the gold file in START and END values, and also the entity type. Only one correct match per entry in the gold file can be matched. Hence, duplicated entries will count as Spurious.
- **Incorrect matches** are reported when START and END values match, but not the type.
- **Partial matches** are reported when two intervals [START, END] have a non-empty intersection, such as the case of “vías respiratorias” and “respiratorias” in the previous example (and matching LABEL). Notice that a partial phrase will only be matched against a single correct phrase. For example, “tipo de cáncer” could be a partial match for both “tipo” and “cáncer”, but it is only counted once as a partial match with the word “tipo”. The word “cancer” is counted then as Missing. This aims to discourage a few large text spans that cover most of the document from getting a very high score.
- **Missing matches** are those that appear in the gold file but not in the dev file.
- **Spurious matches** are those that appear in the dev file but not in the gold file.


### Relation

- **Correct**: relationships that matched exactly to the gold file, including the type and the corresponding IDs for each of the participants.
- **Missing**: relationships that are in the gold file but not in the dev file, either because the type is wrong, or because one of the IDs didn’t match.
- **Spurious**: relationships that are in the dev file but not in the gold file, either because the type is wrong, or because one of the IDs didn’t match.
