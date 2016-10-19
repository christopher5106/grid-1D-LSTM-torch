require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'optim'
local model_utils = require 'model_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple grid lstm 1D network')
cmd:text()
cmd:text('Options')
cmd:option('-hidden',1500,'number of hidden neurons')
cmd:option('-layers',72,'number of layers')
cmd:option('-learning_rate',0.06,'learning rate')
cmd:option('-batch_size',20,'batch size')
cmd:option('-iterations',15000000,'nb of iterations')
cmd:option('-nb_bits',220,'input number of bits')
cmd:option('-eval_interval',1000,'print interval')
cmd:option('-gpu',1,'gpu to use. if 0, cpu')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:text()
opt = cmd:parse(arg)

-- gpu activation
local gpu = opt.gpu > 0
if gpu then
  ok, cunn = pcall(require, 'cunn')
  ok2, cutorch = pcall(require, 'cutorch')
  gpu = gpu and ok and ok2
end

if gpu then
  print("Using GPU", opt.gpu)
  cutorch.setDevice(opt.gpu)
else
  print("GPU OFF")
end

-- build model
cell_dim = opt.hidden

inputs = {}
table.insert(inputs,nn.Identity()())
m0 = inputs[1]
h0 = inputs[1]

forget_lin = nn.Linear(opt.hidden,cell_dim)
input_lin = nn.Linear(opt.hidden,cell_dim)
candidate_lin = nn.Linear(opt.hidden,cell_dim)
output_lin = nn.Linear(opt.hidden,cell_dim)

forget_gate_ = nn.Sigmoid()(forget_lin(h0))
input_gate_ = nn.Sigmoid()(input_lin(h0))
candidate_ = nn.Tanh()( candidate_lin(h0) )
output_gate_ = nn.Sigmoid()(output_lin(h0))

m = nn.CAddTable()( { nn.CMulTable()({input_gate_,candidate_}),nn.CMulTable()({forget_gate_,m0}) })
h = nn.Tanh()( nn.CMulTable()({ output_gate_, m }) )

for n=2,opt.layers do
  forget_lin_ = nn.Linear(opt.hidden,cell_dim)
  forget_lin_:share( forget_lin, 'weight', 'bias','gradWeight', 'gradBias'  )
  forget_gate_ = nn.Sigmoid()(forget_lin_(h))

  input_lin_ = nn.Linear(opt.hidden,cell_dim)
  input_lin_:share( input_lin, 'weight', 'bias','gradWeight', 'gradBias'  )
  input_gate_ = nn.Sigmoid()(input_lin_(h))

  candidate_lin_ = nn.Linear(opt.hidden,cell_dim)
  candidate_lin_:share( candidate_lin, 'weight', 'bias','gradWeight', 'gradBias'  )
  candidate_ = nn.Tanh()(candidate_lin_(h))

  output_lin_ = nn.Linear(opt.hidden,cell_dim)
  output_lin_:share( output_lin, 'weight', 'bias','gradWeight', 'gradBias'  )
  output_gate_ = nn.Sigmoid()(output_lin_(h))

  m = nn.CAddTable()( { nn.CMulTable()({input_gate_,candidate_}),nn.CMulTable()({forget_gate_,m}) })
  h = nn.Tanh()( nn.CMulTable()({ output_gate_, m }) )
end

outputs = {}
table.insert(outputs, nn.Sigmoid()(nn.Linear( cell_dim + opt.hidden , 1)( nn.JoinTable(2)({h,m}) )))

mlp = nn.gModule(inputs, outputs)

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(mlp)

-- initialization
if do_random_init then
  params:uniform(-0.08, 0.08) -- small uniform numbers
end

criterion = nn.BCECriterion()
if gpu then
  mlp = mlp:cuda()
  criterion=criterion:cuda()
end

-- generate data
function next_batch()
  input = torch.zeros(opt.batch_size,opt.hidden)
  input[{{},{1,opt.nb_bits}}] = torch.rand(opt.batch_size,opt.nb_bits):round()
  target = (input:sum(2) % 2 +1):squeeze()
  if gpu then
     input=input:cuda()
     target = target:cuda()
  end
  return input, target
end

function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()

  local input, target = next_batch()
  local out = mlp:forward(input)
  local loss = criterion:forward(out, target)
  local gradOut =criterion:backward(out, target)
  mlp:backward(input, gradOut)
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  return loss, grad_params
end

function eval()
  local input, target = next_batch()
  local out = mlp:forward(input)
  local loss = criterion:forward(out, target)
  return out,loss
end

---------Training:------------
training_losses,accuracies = {},{}
local optim_state = {learningRate = opt.learning_rate}
for e=1,opt.iterations/opt.batch_size do
   optim.adagrad(feval, params, optim_state)
   if e % opt.eval_interval == 0 then
      local out,loss = eval()
      table.insert(training_losses, loss)
      accuracy = (((out[{{},1}]):round()-target):pow(2)):mean()
      table.insert(accuracies,accuracy)
      print("Iteration", e,"Training Loss",loss,"Accuracy",accuracy)
   end
end

---------Saving:------------
torch.save(string.format( "shared_w_accuracies_i%s_lr%s_h%s_lay%s_b%s", opt.iterations, opt.learning_rate, opt.hidden, opt.layers, opt.batch_size), accuracies)
torch.save(string.format( "shared_w_losses_i%s_lr%s_h%s_lay%s_b%s", opt.iterations, opt.learning_rate, opt.hidden, opt.layers, opt.batch_size), training_losses)
