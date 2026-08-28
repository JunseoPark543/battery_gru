from module import * 

############### 모델 전체 구조 #################
class LatentModel(nn.Module):
    def __init__(self, num_hidden,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden)
        self.decoder = Decoder(num_hidden)  
        self.BCELoss = nn.BCELoss() # Binary Cross Entropy Loss  

    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1) 

        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)

        # Training -> posterior z 사용
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior 
        # Inference -> prior z 사용 
        else:
            z = prior 

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        r = self.deterministic_encoder(context_x,context_y,target_x)

        y_pred = self.decoder(target_x, r, z)

        # For training
        if target_y is not None:
            bce_loss = self.BCELoss(t.sigmoid(y_pred), target_y)

            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)

            loss = bce_loss + kl 

        # For Generation 
        else:
            log_p = None
            kl = None
            loss = None

        return y_pred, kl, loss

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div


