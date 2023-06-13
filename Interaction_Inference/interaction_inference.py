import torch


class InteractionInference():
    """
    Interaction inference class.

    An instance of this class provides the functionality to infer interactions.

    Parameters
    ----------
    model : torch.nn.Module
        Recurrent neural network model which might be pretrained.
    initial_model_states : torch.Tensor or tuple
        Initial hidden (and cell) state of the model.
    opt_accessor : function
        Function that returns list of tensors to be optimized alongside of
        the context, usually the hidden state.
    context : torch.Tensor of shape (1, batch, context_size)
        Initial context.
    criterion : function
        Criterion for comparison of a list of past predictions and a list of
        observations.
    optimizer : torch.optim.Optimizer
        Optimizer to optimize the context and hidden state with.
    reset_optimizer : bool
        If True the optimizer's statistics are reset before each inference.
        If False the optimizer's statistics are kept across inferences.
    inference_length : int
        Number of past steps considered during optimization.
    inference_cycles : int
        Number of inference cycles, i.e. how often inference is repeated.
    context_handler : function
        Function that is applied to the context after each optimization,
        e.g. can be used to keep context in certain range.

    """

    def __init__(
            self, model, initial_model_state, 
            opt_accessor, context, criterion,
            optimizer, reset_optimizer=True, inference_length=5,
            inference_cycles=5, context_handler=lambda x: x):

        assert (len(context.shape) ==
                3), "context should be of shape (seq_len, batch, input_size)"
        assert (context.size(0) == 1), "seq_len of context should be 1"
        assert (len(opt_accessor(initial_model_state)) == len(
            optimizer.param_groups[1]['params'])), "opt_accessor must return list that has same length as second second param_group of optimizer"

        self._model = model
        self._model_state = initial_model_state
        self._opt_accessor = opt_accessor
        self._context = context
        self._context.requires_grad = True
        self._criterion = criterion
        self._optimizer = optimizer
        self._reset_optimizer = reset_optimizer
        self._inference_length = inference_length
        self._inference_cycles = inference_cycles
        self._context_handler = context_handler

        if self._reset_optimizer:
            self._optimizer_orig_state = optimizer.state_dict()

        # Buffers to store the histories of inputs and outputs
        self._model_inputs = []
        self._observations = []

        self._model_state = initial_model_state
        for s in self._opt_accessor(self._model_state):
            s.requires_grad_()

        assert (len(self._opt_accessor(self._model_state)) ==
                len(self._optimizer.param_groups[1]['params']))

    def predict(self, state):
        """
        Predict from the past.

        Predict observations given past inputs as well as an initial hidden
        state and a context.

        Parameters
        ----------
        state : torch.Tensor or tuple
            Initial hidden (and cell) state of the network.

        Returns
        -------
        outputs : list
            Result of the prediction.
        states : list of torch.Tensor or list of tuple
            Hidden (and cell) states of the model corresponding to the
            predicted outputs.

        """

        outputs = []
        states = []
        # Forward pass over inference_length steps
        for ci_t in range(len(self._model_inputs)):
            in_t = self._model_inputs[ci_t]
            in_t_c = torch.cat((self._context, in_t), dim=2)
            output, state = self._model.forward(in_t_c, state)
            outputs.append(output)
            states.append(state)
        return outputs, states

    def infer_contexts(self, model_input, observation):
        """
        Optimize the current context.

        Given current inputs and observations, this method infers a context
        based on past model inputs and observations.

        Parameters
        ----------
        model_input : torch.Tensor
            Last input sent to model.
        observation : torch.Tensor
            Last observation made.

        Returns
        -------
        context : torch.Tensor
            Optimized context.
        outputs : list
            Past predictions corresponding to the optimized context and
            possibly hidden and/or cell state.
        states : list of torch.Tensor or list of tuple
            Past model states corresponding to the optimized context and
            possibly hidden and/or cell state.

        """

        assert (len(model_input.shape) ==
                3), "model_input should be of shape (seq_len, batch, input_size)"
        assert (model_input.size(0) == 1), "seq_len of model_input should be 1"
        assert (len(observation.shape) ==
                3), "observation should be of shape (seq_len, batch, input_size)"
        assert (observation.size(0) == 1), "seq_len of observation should be 1"

        if self._reset_optimizer:
            self._optimizer.load_state_dict(self._optimizer_orig_state)

        # Shift inputs and observations by one
        self._model_inputs.append(model_input)
        self._model_inputs = self._model_inputs[-self._inference_length:]
        self._observations.append(observation)
        self._observations = self._observations[-self._inference_length:]

        # Perform context inference cycles
        for _ in range(self._inference_cycles):
            self._optimizer.zero_grad()

            outputs, states = self.predict(self._model_state)

            # Compute loss
            loss = self._criterion(outputs, self._observations)

            # Backward pass
            loss.backward()
            self._optimizer.step()

            # Operations on the data are not tracked
            self._context.data = self._context_handler(self._context.data)

        # Context and state have been optimized; this optimized context/state
        # is now propagated once more in forward direction in order to generate
        # the final output and state to be returned
        with torch.no_grad():
            outputs, states = self.predict(self._model_state)
            for i in range(len(self._model_state)):
                for j in range(len(self._model_state[i])):
                    self._model_state[i][j].data = states[0][i][j].data

        return self._context, outputs, states