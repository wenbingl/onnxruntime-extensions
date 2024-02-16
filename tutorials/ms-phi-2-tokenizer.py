from transformers import AutoTokenizer
from onnxruntime_extensions import gen_processing_models, ort_inference


tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2", trust_remote_code=True, torch_dtype="auto", use_fast=False)

code = '''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """
   primes = []
   for num in range(2, n+1):
       is_prime = True
       for i in range(2, int(math.sqrt(num))+1):
           if num % i == 0:
               is_prime = False
               break
       if is_prime:
           primes.append(num)
   print(primes)```'''

ids = tokenizer(code, return_tensors="np", return_attention_mask=False)
ort_tok, ort_detok = gen_processing_models(tokenizer, pre_kwargs={}, post_kwargs={})
actual_ids, *_ = ort_inference(ort_tok, [code])
print(len(ids['input_ids'].shape), len(actual_ids.shape))
if (ids['input_ids'] != actual_ids).any():
    print('Mismatched results!')
    print(ids['input_ids'], '\n', actual_ids)
else:
    print(ids['input_ids'])

text = ort_inference(ort_detok, actual_ids)
print(text[0])
