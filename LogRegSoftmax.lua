local LogRegSoftMax, _ = torch.class('nn.LogRegSoftMax', 'nn.Module')

function LogRegSoftMax:updateOutput(input)
   return input.nn.LogRegSoftMax_updateOutput(self, input)
end

function LogRegSoftMax:updateGradInput(input, gradOutput)
   return input.nn.LogRegSoftMax_updateGradInput(self, input, gradOutput)
end
