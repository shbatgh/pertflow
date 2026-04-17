import esm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

samd11 = """mpavkkefpg redlalalat fhptlaalpl pplpgylapl paaaalppaa slpasaagye
allapplrpp raylslheaa phlhlprdpl alerfsataa aapdfqplld ngepcievec
ganrallyvr klcqgskgps irhrgewltp nefqfvsgre takdwkrsir hkgkslktlm
skgilqvhpp icdcpgcris spvnrgrlad krtvalpaar nlkkertpsf sasdgdsdgs
gptcgrrpgl kqedgphiri mkrrvhthwd vnisfreasc sqdgnlptli ssvhrsrhlv
mpehqsrcef qrgsleiglr pagdllgkrl grsprissdc fsekrarses pqealllpre
lgpsmapedh yrrlvsalse astfedpqrl yhlglpshdl lrvrqevaaa alrgpsglea
hlpsstagqr rkqglaqhre gaapaaapsf serelpqppp llspqnaphv algphlrppf
lgvpsalcqt pgygflppaq aemfawqqel lrkqnlarle lpadllrqke lesarpqlla
petalrpndg aeelqrrgal lvlnhgaapl lalppqgppg sgpptpsrds arraprkggp
gpasarpses kemtgarlwa qdgsedeppk dsdgedpeta avgcrgptpg qapaggagae
gkglfpgstl plgfpyavsp yfhtgavggl smdgeeapap edvtkwtvdd vcsfvgglsg
cgeytrvfre qgidgetlpl lteehlltnm glklgpalki raqvarrlgr vfyvasfpva
lplqpptlra perelgtgeq plspttatsp yggghalagq tspkqengtl allpgapdps
qplc"""

protein = ESMProtein(sequence=samd11)
client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"
protein_tensor = client.encode(protein)
logits_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
print(logits_output.logits, logits_output.embeddings)