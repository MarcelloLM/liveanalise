import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objs as go
import plotly.io as pio
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL                 import Image

# Verificar e baixar pacotes necessários
def verificar_downloads_nltk():
    pacotes = ['punkt', 'stopwords']
    for pacote in pacotes:
        try:
            nltk.data.find(f'tokenizers/{pacote}')
        except LookupError:
            nltk.download(pacote)

verificar_downloads_nltk()

# Carregar dados
caminho_arquivo = r"C:\Users\46241887807\Desktop\Py\CategorizacaoMensagens\bases\Conversas.csv"
try:
    Conversas = pd.read_csv(caminho_arquivo)
    Conversas['startTime'] = pd.to_datetime(Conversas['startTime'])
except FileNotFoundError:
    st.write(f"O arquivo CSV não foi encontrado em {caminho_arquivo}. Verifique o caminho.")
except Exception as e:
    st.write(f"Ocorreu um erro ao carregar o arquivo CSV: {e}")

# Função para adicionar stopwords personalizadas
def obter_stopwords_personalizadas():
    stopwords_personalizadas = [
        'dia', 'saber', 'preciso', 'fazer', 'bem', 'ainda', 'pra', 'tarde',
                      'tudo', 'gostaria', 'alex', 'contato', 'sobre', 'favor', 'vou',
                      'obrigado', 'consigo', 'pois', 'bom', 'fiz', 'recebi', 'obrigada',
                      'agora', 'boa', 'aluno', 'faço', 'pode', 'caso', 'ter', 'posso',
                      'abri', 'ajudar', 'aparece', 'então', 'certo', 'varias', 'vezes',
                      'acabamos', 'viajando', 'peguei', 'nela', 'ex', 'refizessem', 'brasil',
                      'fala', 'sei', 'veio', 'enviei', 'perguntar', 'portanto', 'pq', 'realmente',
                      'sendo', 'forma', 'precisei', 'terminar', 'utilizar', 'world', 'solicitação',
                      'ano', 'site', 'solicitar', 'algum', 'semestre', 'outra', 'aqui', 'semana',
                      'porém', 'enviar', 'porém', 'outra', 'passado', 'aqui', 'falar', 'nao',
                      'sim', 'queria', 'vcs', 'vai', 'dias', 'envio', 'quero', 'porque', 'tempo',
                      'hoje', 'referente', 'possível', 'consegui', 'vc', 'verificar', 'apenas',
                      'quanto', 'paguei', 'desde', 'alguma', 'assim', 'consegue', 'poderia',
                      'ja', 'chamado', 'resposta', 'ver', 'duas', 'email', 'mim', 'único',
                      'cursar', 'resolver', 'realizar', 'ajuda', 'seguir', 'nada', 'mês',
                      'gentileza', 'antes', 'dúvida', 'aí', 'tentando', 'disponível',
                      'mandar', 'faz', 'consta', 'solicitei', 'nenhum', 'ir', 'abrir',
                      'noite', 'abrir', 'aguardo', 'pedir', 'outro', 'acesso', 'informar',
                      'momento', 'conta', 'desse', 'entrar', 'final', 'janeiro', 'informações',
                      'aberto', 'assinatura', 'pago', 'início', 'onde', 'turma', 'sabe', 'dizer',
                      'troquei', 'nbsp', 'alun', 'digite', 'menu', 'voltar', 'anterior',
                      'escolha', 'informe', 'pesquisa', 'ótimo', 'correspondente', 'regular',
                      'alun', 'irá', 'satisfação', 'breve', 'aguarde', 'nome', 'volta', 'confirmar',
                      'breve', 'especialistas', 'assunto', 'deseja', 'agilizar', 'abaixo', 'opções',
                      'abaixo', 'ruim', 'número', 'avalia', 'participe', 'melhorar', 'olá', 'descreva',
                      'https', 'link', 'felizes', 'agradecemos', 'melhorar', 'ajude', 'tendimento',
                      'péssimo', 'conversar', 'gt', 'lgpd', 'através', 'leia', 'conosco', 'concordo',
                      'discordo', 'agradece', 'proteção', 'ficamos', 'interesse', 'oi', 'enviando',
                      'qualquer', 'volte', 'obrigad', 'registo', 'atendimento', 'atender', 'mensagem',
                      'whatsapp', 'geral', 'lei', 'expirou', 'legais', 'termos', 'cpf', 'família',
                      'incríveis', 'garantimos', 'algo', 'ajudo', 'desejo', 'atendid', 'dados', 'nov',
                      'situações', 'vamos', 'perguntas', 'algumas', 'disposição', 'diga', 'assistente',
                      'fecap', 'pontos', 'ajudo', 'algo', 'desejo', 'especialista', 'atendid', 'vamos',
                      'algumas', 'nov', 'assistente', 'família', 'assistente', 'garantimos', 'momentos',
                      'from', 'to', 'subject', 're', 'diversas', 'virtual', 'tire', 'dúvidas', 'desejada',
                      'atividade', 'combinado', 'próxima', 'álvares', 'liberdade', 'penteado', 'opção',
                      'oferecer', 'fundação', 'escola', 'comércio', 'basta', 'estarei', 'elevado', 'demanda',
                      'ok', 'relação', 'feedback', 'conte', 'sempre', 'comigo', 'precisar', 'chamo', 'alta',
                      'espera', 'pedimos', 'desculpa', 'compreensão', 'desculpas', 'relacionamento', 'agradeço',
                      'inatividade', 'encerrar', 'precisar', 'nova', 'atenciosamente', 'financeiro', 'lt',
                      'lg', 'penteado', 'requerimentos', 'abertura', 'entendi', 'desculpe', 'comentário',
                      'deixe', 'exceto', 'feriados', 'pinheiros', 'penteadoliberdade', 'fecapfecap', 'acessar',
                      'área', 'escreveu', 'asafecap', 'you', 'if', 'rodrigo', 'zamora', 'damião', 'saúde', 'espero',
                      'centro', 'bookings', 'http', 'av', 'universitário', 'freitas', 'ronaldo', 'ribeiro', 'abr',
                      'maria', 'lurdes', 'regularize', 'situação', 'evite', 'zamora', 'michele', 'brasilia',
                      'document', 'the', 'disso', 'além', 'sender', 'ios', 'lbarragan', 'prezado', 'cambria',
                      'marcelo', 'message', 'this', 'or', 'termo', 'compromisso', 'about', 'questions', 'share',
                      'not', 'jun', 'alvarista', 'sucesso', 'tesouraria', 'enviado', 'abril', 'problemas', 'futuros',
                      'março', 'precisa', 'pmto', 'april', 'sucessoalvarista', 'gestão', 'signing', 'with', 'casimiro',
                      'lorena', 'unsubscribe', 'paulo', 'pedro', 'joão', 'manter', 'sabemos', 'suporte', 'precisando',
                      'parcela', 'existe', 'concluído', 'submetido', 'meio', 'lembrar', 'fluxo', 'confira', 'toda',
                      'comunidade', 'exibir', 'maio', 'preparamos', 'dicas', 'clique', 'promover', 'educação',
                      'fácil', 'tarefa', 'henrique', 'kaique', 'responda', 'financeira', 'busca', 'queremos', 'contas',
                      'verificamos', 'instituto', 'kaylaine', 'glenda', 'segunda', 'fecapav', 'iifecap', 'jr',
                      'crédito', 'cartão', 'at', 'apr', 'bate', 'papo', 'android', 'cópia', 'coloco', 'setor',
                      'responsável', 'renata', 'bianca', 'reserva', 'caixa', 'correio', 'ref', 'tim', 'claro',
                      'oliveira', 'vailson', 'nº', 'vailsonalmeida', 'aluna', 'claro', 'indevida', 'prezados',
                      'entanto', 'any', 'tel', 'oliveirarg', 'caroline', 'nenhuma', 'deixou', 'vez', 'desconhecia',
                      'retorno', 'anexo', 'segue', 'zago', 'matheus', 'valor', 'total', 'fornereto', 'procurados',
                      'fabricio', 'identificada', 'solicitamos', 'cobrança', 'vencimento', 'data', 'após','enviada',
                      'junho', 'nadin', 'tabata', 'quase', 'cheia', 'custodio', 'beatriz', 'lado', 'abra', 'senha',
                      'ra', 'portal', 'quais', 'todos', 'passo', 'cc', 'sejafecap', 'secretaria', 'selecione', 'fiescomissão',
                      'cpsa', 'acompanhamentofecap', 'tela', 'selecione', 'equipe', 'asa', 'canal', 'informativo',
                      'anexe', 'digitalizados', 'pdf', 'atenção', 'possui', 'guilherme', 'gomes', 'efetuar', 'pagamento',
                      'esquerdo', 'faltantes', 'campo', 'requerimento', 'permanente', 'sent', 'carvalho', 'santos',
                      'silva', 'vaz', 'sabrina', 'pmpara', 'august', 'hengles', 'rafael', 'acima', 'primeira',
                      'paciente', 'brt', 'carmen', 'continuar', 'gente', 'pessoa', 'cnpj', 'andressa', 'rodrigues',
                      'podem', 'clientes', 'olhada', 'detalhes', 'new', 'booking', 'venha', 'parte', 'tue', 'sep',
                      'cancer', 'center', 'conseguir', 'comparecer', 'avise', 'wed', 'date', 'oct', 'thu', 'começar',
                      'primeiro', 'server', 'returned', 'generating', 'failed', 'these', 'has', 'headers', 'ask',
                      'them', 'recipient', 'by', 'phone', 'some', 'other', 'status', 'code', 'information', 'see',
                      'deliver', 'unable', 'can', 'fix', 'repeated', 'attempts', 'that', 'their', 'requests',
                      'connection', 'see', 'contact', 'your', 'and', 'tell', 'delivery', 'id', 'system', 'dê',
                      'smtp', 'recipients', 'group', 'admin', 'is', 'means', 'example', 'admin', 'likely', 'adm',
                      'only', 'is', 'account', 'tried', 'reach', 'did', 'it', 'despite', 'because', 'expired',
                      'who', 'more', 'was', 'administrators', 'gsmtp', 'before', 'but', 'crlf', 'please',
                      'try', 'in', 'over', 'one', 'in', 'accepting', 'delivered', 'quota', 'please', 'parque',
                      'sintonia', 'atlântica', 'rocha', 'mailbox', 'full', 'rocha', 'condominial', 'received',
                      'entregar', 'amazonses', 'delete', 'retype', 'groups', 'entregar', 'frente', 'alvaristafecap',
                      'area', 'esperando', 'joao', 'spoladore', 'concorrer', 'escolhe', 'ta', 'útil', 'escolhendo',
                      'alguns', 'ficar', 'pontes', 'jackson', 'natália', 'fontenele', 'tibério', 'garcia', 'continuidade',
                      'dar', 'luiz', 'vitória', 'lourenco', 'pinha', 'correa', 'eduardo', 'poderiam', 'levar', 'enzo',
                      '메일에서 발송된', '발송된', '메일에서', 'martinez', 'fabiana', 'permanecemos', 'fabiana',
                      'informação', 'úteis', 'esqueça', 'aba', 'tipo', 'olha', 'confirmaremos', 'lá', 'tirar',
                      'prontinho', 'aguarde', 'alternativa', 'única', 'gerando', 'justamente', 'entraremos',
                      'contigo', 'meses', 'colei', 'esclarecer', 'aryane', 'monteiro', 'dá', 'errônea',
                      'urgente', 'selecionado', 'esclusivamente', 'dessa', 'erros', 'escrita', 'usa', 'má',
                      'fé', 'parece', 'piada', 'absurdo', 'usa', 'legislação', 'vigente', 'recebeu', 'divulgação',
                      'filho', 'peço',  'necessário', 'media', 'redirecionar', 'errei', 'iam', 'errado', 'errando',
                      'att', 'errada', 'informaram', 'virar', 'desanime', 'atento', 'fique', 'iii', 'security',
                      'simples', 'veja', 'ninguém', 'atende', 'obriga', 'tutorial', 'sido', 'havia', 'deverá',
                      'digamos', 'abchief', 'office', 'imediatamente', 'leitura', 'utilização', 'sinto',
                      'todas', 'tratar', 'moreira', 'fernando', 'sérgio', 'jhonatan', 'jhowmsilva', 'gustavo',
                      'andrade', 'dada', 'largada', 'jm', 'paraviso', 'geovani', 'albino', 'valeria', 'wednesday',
                      'misseno', 'lucas', 'january', 'uso', 'terceiro', 'rogério', 'romrsouza', 'mendonça', 'lima',
                      'mousadis', 'stephanymousadis', 'maciel', 'mendonça', 'davi', 'reais', 'mil', 'fróes',
                      'francisco', 'rfcarvalho', 'cruz', 'nogueira', 'stephany', 'nathan', 'ivacostafag', 'costa',
                      'julia', 'castro', 'fagundes', 'eduarda', 'aguarda', 'mendes', 'removed', 'been', 'fale',
                      'interação', 'vanderley', 'pereira', 'vanderley', 'head', 'marketing', 'growth', 'camila',
                      'barbosa', 'josinaldo', 'pereira', 'fw', 'vanderleypereira', 'encaminhada', 'meios', 'emilly',
                      'mickeli', 'mickeli', 'depine', 'giovanna', 'campos', 'assistência', 'deliza', 'badaró', 'finco',
                      'perdoe', 'demora', 'alves', 'noriangelys', 'andreina', 'gi', 'barroso', 'smart', 'tv',
                      'luishenriquealegriadefreitas', 'thomé', 'marques', 'thomé', 'mejias', 'zapata', 'bruno', 'milano',
                      'bmilanob', 'docs', 'forwarded', 'neste', 'deve', 'ementa', 'precisaria', 'desses', 'gabriella',
                      'natali', 'felipeandrade', 'ana', 'virginia', 'bortolai', 'mathias', 'filha', 'laura', 'canella',
                      'lauracanella', 'accademic', 'dahan', 'ifrs', 'usgaap', 'programme', 'enterprise', 'risk',
                      'management', 'food', 'risk', 'chief', 'officer', 'natasha', 'borali', 'regards', 'jijo','gabriel',
                      'acerca', 'razão', 'director', 'cristiny', 'erro', 'ocorreu', 'poderemos', 'casa', 'torna',
                      'arruda', 'luna', 'suzana', 'adriane', 'bazi', 'carolina', 'constantino', 'letícia', 'surgindo',
                      'alessandra', 'recebemos', 'willian', 'araujo', 'ligo', 'mando', 'boas', 'vindas', 'abre', 'indignada',
                      'receber', 'querem', 'responde', 'hr', 'help', 'hj', 'segundas', 'respeite', 'min', 'escolhido',
                      'ativo', 'deste', 'deletícia', 'recebendo', 'notificação', 'assinei', 'aplicam', 'daniely', 'lenares',
                      'thursday', 'marie', 'february', 'stephaniemvicente', 'stéphanie', 'patrick', 'santana',
                      'souza', 'carolinacatelli', 'sanchez', 'patricksanttana', 'internada', 'mãe', 'hospital', 'felipe',
                      'ruiz', 'mayara', 'soares', 'falaram', 'participar', 'matta', 'walcher', 'transaction', 'digital', 'obter',
                      'outlook', 'need', 'enter', 'docusignsign', 'solution', 'read', 'service', 'declining', 'signature',
                      'even', 'home', 'hrhelp', 'assinar', 'may', 'receive', 'powered', 'emailthis', 'modify', 'have', 'emailreport',
                      'sign', 'provides', 'professional', 'review','docusign', 'trusted', 'whether', 'an', 'click', 'documents',
                      'secure', 'access', 'across', 'binding', 'managing', 'are', 'receiving', 'reminder', 'rather', 'would',
                      'gabrielle', 'contains', 'business', 'english', 'fevereiro', 'ciências', 'contábeis',
                      'methodvisit', 'services', 'electronic', 'request', 'page', 'on', 'just', 'minutes',
                      'visit', 'our', 'safe', 'electronically', 'support', 'app', 'details', 'globe', 'ensino',
                      'médio', 'legally', 'out', 'emailing', 'having', 'trouble', 'using', 'relações', 'internacionais',
                      'colação', 'grau', 'curso', 'pós', 'ola', 'okk', 'ola ola', 'publicidade', 'propaganda', 'atendente',
                      'fundamentos', 'finanças', 'tópico', 'inferencial', 'inferencial', 'ltsucessoalvaristafecapbrgt',
                      'set', 'alvaristafecapnbsp', 'fecapbr', 'stefany', 'sousa', 'setembro', 'quartafeira', 'abraços',
                      'natalia', 'matos', 'elias', 'dutra', 'sex', 'passando', 'indisponibilidade', 'encontrome',
                      'sousaauxiliar', 'administrativo', 'saiba', 'trabalhando', 'cada', 'trabalhando', 'melhorálo',
                      'infelizmente', 'sistêmica', 'conforme', 'impossibilidade', 'orientado', 'função', 'serviços',
                      'consulta', 'outros', 'sucessoalvaristafecapbr', 'ito', 'cristina', 'comunicado', 'lamentamos',
                      'experiências', 'emba', 'mkt', 'convites', 'remanescentes', 'thelma', 'nbspn...', 'nbspnbspnbspnbspnbspnbspnbspnbspnbspnbsp',
                      'br', '2222', 'hotmail', '55', '11', 'gmail', '3272', '13', '2024', 'quarta', 'sexta', 'lo', 'eventuais', 'transtornos',
                      'melhorá', 'causados', 'terça', 'feira', '' 
        # Adicione todas as palavras que você listou aqui
    ]
    return set(stopwords.words('portuguese') + stopwords_personalizadas)

stopwords_completas = obter_stopwords_personalizadas()

# Função para pré-processamento de texto
def preprocessar_texto(texto, stopwords):
    if not isinstance(texto, str):
        return ""  # Se não for string, retorna string vazia
    texto_limpo = re.sub(r'\W+', ' ', texto.lower())  # Converte para minúsculas e remove caracteres especiais
    tokens = word_tokenize(texto_limpo)  # Tokeniza o texto
    tokens_sem_stopwords = [token for token in tokens if token not in stopwords]  # Remove stopwords
    return ' '.join(tokens_sem_stopwords)  # Junta os tokens de volta em um texto


# Verifique se a coluna 'mensagem' existe
if 'transcriptConsumer' in Conversas.columns:
    # Aplicar pré-processamento no dataset
    Conversas['transcriptConsumer_clean'] = Conversas['transcriptConsumer'].apply(lambda x: preprocessar_texto(x, stopwords_completas))
    # st.write(Conversas.head())

    # Função para gerar wordcloud
    def gerar_wordcloud(texto):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        st.pyplot(plt)

    # Gerar wordcloud com todas as mensagens limpas
    st.title("Word Cloud")
    texto_todas_mensagens = ' '.join(Conversas['transcriptConsumer_clean'])
    gerar_wordcloud(texto_todas_mensagens)

else:
    st.write("Coluna 'mensagem' não encontrada no DataFrame.")


# Ajustar e transformar os dados
vectorizer = CountVectorizer(ngram_range=(2, 2))
mensagens_processadas = Conversas['transcriptConsumer_clean']
TransformarMensagens = vectorizer.fit_transform(mensagens_processadas)

vocabulario = vectorizer.get_feature_names_out()
frequencia = TransformarMensagens.sum(axis=0)

dados_frequencia = pd.DataFrame({'Palavra': vocabulario, 'Frequencia': frequencia.flat})
dados_frequencia = dados_frequencia.sort_values(by='Frequencia', ascending=False)

# Análise de frequência por mês
Conversas['mes_ano'] = Conversas['startTime'].dt.to_period('M')
frequencia_por_mes = pd.DataFrame(columns=['Palavra', 'Frequencia', 'mes_ano'])
for mes in Conversas['mes_ano'].unique():
    mensagens_mes = Conversas[Conversas['mes_ano'] == mes]['transcriptConsumer_clean']
    TransformarMensagens_mes = vectorizer.transform(mensagens_mes)
    frequencia_mes = TransformarMensagens_mes.sum(axis=0)
    dados_frequencia_mes = pd.DataFrame({'Palavra': vocabulario, 'Frequencia': frequencia_mes.flat})
    dados_frequencia_mes['mes_ano'] = mes
    frequencia_por_mes = pd.concat([frequencia_por_mes, dados_frequencia_mes], ignore_index=True)

Conversas['Hora'] = Conversas['startTime'].dt.hour
Conversas['Dia_Semana'] = Conversas['startTime'].dt.dayofweek
dias_semana = {0: 'Segunda-feira', 1: 'Terça-feira', 2: 'Quarta-feira', 3: 'Quinta-feira', 4: 'Sexta-feira', 5: 'Sábado', 6: 'Domingo'}
Conversas['Dia_Semana'] = Conversas['Dia_Semana'].map(dias_semana)
conversas_por_hora = Conversas.groupby('Hora').size().reset_index(name='Quantidade')
conversas_por_hora = conversas_por_hora.sort_values(by='Hora')
conversas_por_dia = Conversas.groupby('Dia_Semana').size().reset_index(name='Quantidade')
conversas_por_dia['Dia_Semana'] = pd.Categorical(conversas_por_dia['Dia_Semana'], categories=['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo'], ordered=True)
conversas_por_dia = conversas_por_dia.sort_values(by='Dia_Semana')

# Mapeamento de termos para grupos de temas
mapeamento_temas = {
    'bilhete estudante': 'bilhete estudante',
    'sp trans': 'bilhete estudante',
    'passe livre': 'bilhete estudante',
    'passe escolar': 'bilhete estudante',
    'revalidar bilhete': 'bilhete estudante',
    'começam aulas': 'inicio aulas',
    'aulas começam': 'inicio aulas',
    'historico escolar': 'historico escolar',
    'histórico escolar': 'historico escolar',
    'bolsa prouni': 'bolsa prouni',
    'bolsita prouni': 'bolsa prouni',
    'boleto mensalidade' : 'boleto mensalidade',
    'boleto rematricula' : 'boleto mensalidade',
    'boleto rematrícula' : 'boleto mensalidade',
    'pagar boleto' : 'boleto mensalidade',
    'pagar mensalidade' : 'boleto mensalidade',
    'pagar rematrícula' : 'boleto mensalidade',
    'rematrícula boleto' : 'boleto mensalidade'
}

def mapear_tema(bigrama):
    for termo, grupo in mapeamento_temas.items():
        if termo in bigrama:
            return grupo
    return bigrama  # Se não houver mapeamento, retorna o próprio bigrama

frequencia_por_mes['Grupo_Tema'] = frequencia_por_mes['Palavra'].apply(mapear_tema)
frequencia_por_mes['Frequencia'] = pd.to_numeric(frequencia_por_mes['Frequencia'])
frequencia_agrupada = frequencia_por_mes.groupby(['Grupo_Tema', 'mes_ano'])['Frequencia'].sum().reset_index()

top_10_temas = frequencia_agrupada.groupby('Grupo_Tema')['Frequencia'].sum().nlargest(10).index
frequencia_top_10 = frequencia_agrupada[frequencia_agrupada['Grupo_Tema'].isin(top_10_temas)]
dados_frequencia['Grupo_Tema'] = dados_frequencia['Palavra'].apply(mapear_tema)
top_10_frequencia = dados_frequencia.groupby('Grupo_Tema')['Frequencia'].sum().nlargest(10).reset_index()

def analisar_bigrams_para_grupo(dados_frequencia, grupo_tema):
    dados_grupo_especifico = dados_frequencia[dados_frequencia['Grupo_Tema'] == grupo_tema]
    frequencia_bigramas_grupo = dados_grupo_especifico.groupby('Palavra')['Frequencia'].sum().reset_index()
    frequencia_bigramas_grupo = frequencia_bigramas_grupo.sort_values(by='Frequencia', ascending=False)
    return frequencia_bigramas_grupo

# Aplicação Streamlit
st.title("Análise de Conversas")
st.write("### Frequência de Bigrams por Tema")

# Carregar e mostrar a imagem na barra lateral
image = Image.open(r"C:\Users\46241887807\Desktop\Py\CategorizacaoMensagens\img\Logo OFC V.2.png")  # Use r antes do caminho para tratar como string raw
st.sidebar.image(image)

# Seção de filtro de data
st.sidebar.header('Filtros de Data')
data_inicio = st.sidebar.date_input('Data de Início', value=pd.to_datetime(Conversas['startTime']).min())
data_fim = st.sidebar.date_input('Data de Fim', value=pd.to_datetime(Conversas['startTime']).max())
Conversas_filtradas = Conversas[(Conversas['startTime'] >= pd.to_datetime(data_inicio)) & (Conversas['startTime'] <= pd.to_datetime(data_fim))]


# Filtrando os dados com base nas datas selecionadas
frequencia_por_mes_filtrada = frequencia_por_mes[frequencia_por_mes['mes_ano'].isin(Conversas_filtradas['mes_ano'].unique())]
frequencia_agrupada_filtrada = frequencia_por_mes_filtrada.groupby(['Grupo_Tema', 'mes_ano'])['Frequencia'].sum().reset_index()
top_10_temas_filtrados = frequencia_agrupada_filtrada.groupby('Grupo_Tema')['Frequencia'].sum().nlargest(10).index
frequencia_top_10_filtrada = frequencia_agrupada_filtrada[frequencia_agrupada_filtrada['Grupo_Tema'].isin(top_10_temas_filtrados)]

# Gráfico 1: Frequência dos top 10 temas filtrados
fig1, ax1 = plt.subplots()
sns.barplot(x='Frequencia', y='Grupo_Tema', data=frequencia_top_10_filtrada, ax=ax1)
ax1.set_xlabel('Frequência')
ax1.set_ylabel('Temas')
ax1.set_title('Top 10 Temas mais Frequentes (Filtrado)')
st.pyplot(fig1)

# Seção de seleção de tema para detalhar
tema_selecionado = st.selectbox('Selecione um Tema para Detalhar', top_10_temas_filtrados)
bigrams_tema_selecionado = analisar_bigrams_para_grupo(dados_frequencia, tema_selecionado)

# Gráfico 2: Detalhe dos bigrams para o tema selecionado
fig2, ax2 = plt.subplots()
sns.barplot(x='Frequencia', y='Palavra', data=bigrams_tema_selecionado.head(10), ax=ax2)
ax2.set_xlabel('Frequência')
ax2.set_ylabel('Bigrams')
ax2.set_title(f'Top 10 Bigrams para o Tema: {tema_selecionado}')
st.pyplot(fig2)

st.write("### Análise de Frequência de Conversas por Hora do Dia")

# Gráfico 3: Frequência de conversas por hora
fig3, ax3 = plt.subplots()
sns.lineplot(x='Hora', y='Quantidade', data=conversas_por_hora, marker='o', ax=ax3)
ax3.set_xlabel('Hora do Dia')
ax3.set_ylabel('Quantidade de Conversas')
ax3.set_title('Frequência de Conversas por Hora do Dia')
st.pyplot(fig3)

st.write("### Análise de Frequência de Conversas por Dia da Semana")

# Gráfico 4: Frequência de conversas por dia da semana
fig4, ax4 = plt.subplots()
sns.barplot(x='Dia_Semana', y='Quantidade', data=conversas_por_dia, ax=ax4)
ax4.set_xlabel('Dia da Semana')
ax4.set_ylabel('Quantidade de Conversas')
ax4.set_title('Frequência de Conversas por Dia da Semana')
st.pyplot(fig4)
