import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from engression import engression
from src.utils import (
    EarlyStopping,
    train,
    train_no_early_stopping,
    train_trans,
    train_trans_no_early_stopping,
)

from rtdl_revisiting_models import MLP, ResNet, FTTransformer


class _TorchBase:
    def __init__(self, batch_size: int = 32, classification: bool = False, learning_rate: float = 1e-3, weight_decay: float = 0.0, n_epochs: int = 100, patience: int = None, checkpoint_path: str = None, seed: int = None):
        self.batch_size     = batch_size
        self.classification = classification
        self.learning_rate   = learning_rate
        self.weight_decay   = weight_decay
        self.n_epochs       = n_epochs
        self.patience       = patience
        self.checkpoint_path = checkpoint_path
        self.seed           = seed
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare(self, X, y=None):
        Xa = X.values if hasattr(X, "values") else np.asarray(X)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        Xt = torch.tensor(Xa, dtype=torch.float32)
        if y is None:
            return Xt
        ya = y.values if hasattr(y, "values") else np.asarray(y)
        if self.classification:
            yt = torch.tensor(ya.reshape(-1), dtype=torch.long)
            self.n_classes = int(torch.unique(yt).numel())
        else:
            yt = torch.tensor(ya.reshape(-1, 1), dtype=torch.float32)
        return Xt, yt

    def _to_device(self, *tensors):
        return [t.to(self.device) for t in tensors]

    def _loader(self, Xt, yt, shuffle: bool, drop_last: bool = False):
        ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)

    def prepare_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        X_tr_t, y_tr_t = self._prepare(X_train, y_train)
        

        if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
            y_tr_t = y_tr_t.float().view(-1, 1)

        tensors_to_move = [X_tr_t, y_tr_t]

        if X_val is not None and y_val is not None:
            X_va_t, y_va_t = self._prepare(X_val, y_val)
            if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
                y_va_t = y_va_t.float().view(-1, 1)
            tensors_to_move.extend([X_va_t, y_va_t])
        else:
            X_va_t, y_va_t = None, None
            
        if X_test is not None and y_test is not None:
            X_te_t, y_te_t = self._prepare(X_test, y_test)
            if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
                y_te_t = y_te_t.float().view(-1, 1)
            tensors_to_move.extend([X_te_t, y_te_t])
        else:
            X_te_t, y_te_t = None, None

        moved_tensors = self._to_device(*tensors_to_move)
        it = iter(moved_tensors)
        
        X_tr_t, y_tr_t = next(it), next(it)
        if X_va_t is not None: X_va_t, y_va_t = next(it), next(it)
        if X_te_t is not None: X_te_t, y_te_t = next(it), next(it)

        train_loader = self._loader(X_tr_t, y_tr_t, shuffle=True, drop_last=True)
        val_loader = self._loader(X_va_t, y_va_t, shuffle=False, drop_last=True) if X_va_t is not None else None
        test_loader = self._loader(X_te_t, y_te_t, shuffle=False) if X_te_t is not None else None
        

        self.d_in  = X_tr_t.size(1)

        return train_loader, val_loader, test_loader




# ----- MLP Regressor -----
class MLPRegressor(_TorchBase):
    """Feed‐forward MLP for regression with early‐stopping and uncertainty estimation."""
    def __init__(
        self,
        n_blocks: int = 2,
        d_block: int = 128,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        patience: int = 10,
        checkpoint_path: str = "checkpoint_mlp.pt",
        seed: int = None,
        n_epochs: int=100
    ):
        super().__init__(
            batch_size=batch_size,
            classification=False,
            learning_rate=learning_rate,
            weight_decay=weight_decay,     
            n_epochs=n_epochs,             
            patience=patience,             
            checkpoint_path=checkpoint_path, 
            seed=seed                      
        )
        self.n_blocks = n_blocks 
        self.d_block = d_block   
        self.dropout = dropout   
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        Xt, yt = self._prepare(X_train, y_train)
        Xt, yt = self._to_device(Xt, yt)
        train_loader = self._loader(Xt, yt, shuffle=True)

        if X_val is not None and y_val is not None:
            Xv, yv = self._prepare(X_val, y_val)
            Xv, yv = self._to_device(Xv, yv)
            val_loader = self._loader(Xv, yv, shuffle=False)
            early_stop = EarlyStopping(
                patience=self.patience,
                path=self.checkpoint_path
            )
        else:
            val_loader = None
            early_stop = None

        self.d_in = Xt.size(1)
        self.model = MLP(
            d_in=self.d_in,
            d_out=1,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        actual_epochs_run = 0
        if val_loader is not None and self.patience is not None and self.patience > 0:
            early_stop = EarlyStopping(
                patience=self.patience,
                path=self.checkpoint_path
            )
            actual_epochs_run = train(
                self.model,
                criterion,
                optimizer,
                self.n_epochs, # Max epochs for this training run
                train_loader,
                val_loader,
                early_stop,
                self.checkpoint_path
            )
        else:
            actual_epochs_run = train_no_early_stopping(
                self.model,
                criterion,
                optimizer,
                self.n_epochs, # Max epochs for this training run
                train_loader
            )
        return actual_epochs_run

    def predict(self, X):
        Xt = self._prepare(X)
        Xt = Xt.to(self.device)
        dummy_y = torch.zeros((Xt.size(0), 1), dtype=torch.float32, device=self.device)
        loader = self._loader(Xt, dummy_y, shuffle=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                out = self.model(xb).reshape(-1)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds)
    
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X)  # numpy array, shape (N,)

        self.model.eval()
        residuals = []
        with torch.no_grad():
            for xb, yb in train_loader:
                out = self.model(xb.to(self.device)).view(-1).cpu().numpy()
                residuals.append(yb.cpu().numpy().ravel() - out)
        sigma = float(np.std(np.concatenate(residuals), ddof=1))

        if sample_size:
            samples = self.predict_samples(X, sample_size)
            return mu, sigma, samples

        return mu, sigma

    

# ----- MLP Classifier -----
class MLPClassifier(_TorchBase):
    def __init__(self,num_classes: int,  n_blocks=2, d_block=128, dropout=0.5, batch_size=32, learning_rate=1e-3, weight_decay=0.0, n_epochs=100, patience=10, checkpoint_path="checkpoint_mlp.pt", seed=None):
        super().__init__(
    batch_size=batch_size,
    classification=True,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    n_epochs=n_epochs,         
    patience=patience,         
    checkpoint_path=checkpoint_path, 
    seed=seed                  
)
        self.n_blocks = n_blocks 
        self.d_block = d_block   
        self.dropout = dropout 
        self.num_classes = num_classes  
        self.d_out = num_classes
        self.model = None


    def build_model(self):
        if self.model is None:
            if not hasattr(self, 'd_in') or not hasattr(self, 'd_out'):
                raise ValueError("d_in and d_out not set. Call prepare_data first")
            self.model = MLP(

                d_in=self.d_in,
                d_out=self.d_out,
                n_blocks=self.n_blocks,
                d_block=self.d_block,   
                dropout=self.dropout
            ).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed);torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        if self.device.type == "cuda":
            print(f"MLPClassifier: using GPU (device={self.device})")
        else:
            print("MLPClassifier: running on CPU")

        self.build_model()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()
        actual_epochs_run = 0
        if val_loader is not None and self.patience is not None and self.patience > 0:
            early_stop = EarlyStopping(patience=self.patience, path=self.checkpoint_path)
            actual_epochs_run = train(
                self.model, criterion, optimizer,
                self.n_epochs, train_loader, val_loader,
                early_stop, self.checkpoint_path
            )
        else:
            actual_epochs_run = train_no_early_stopping(
                self.model, criterion, optimizer,
                self.n_epochs, train_loader
            )
        return actual_epochs_run

    def predict_proba(self, X):
        Xt = self._prepare(X)
        Xt = Xt.to(self.device)
        loader = DataLoader(TensorDataset(Xt, torch.zeros(len(Xt))), batch_size=self.batch_size, shuffle=False)
        probs = []
        self.model.eval()
        with torch.no_grad():
            for Xb, _ in loader:
                logits = self.model(Xb.to(self.device))
                if self.d_out == 1: # Binary
                    probs_batch = torch.sigmoid(logits.view(-1))
                else: # Multiclass
                    probs_batch = torch.softmax(logits, dim=1)
                probs.append(probs_batch.cpu().numpy())
        return np.concatenate(probs, axis = 0)
    def predict(self, X):
        probabilities = self.predict_proba(X)
        if self.d_out == 1:
            return (probabilities >= 0.5).astype(int)
        else:
            return np.argmax(probabilities, axis=1)
    




class EngressionRegressor(_TorchBase):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_epochs: int = 500,
        num_layer: int = 3,
        hidden_dim: int = 128,
        resblock: bool = False,
        batch_size: int = 32,
        seed: int = None,
        device: str = None,
    ):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate)
        self.num_epochs = num_epochs
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.resblock = resblock
        self.seed = seed
        self.device = torch.device(device) if device else self.device
        self.model = None

    def fit(self, X_train, y_train):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        Xt, yt = self._prepare(X_train, y_train)
        Xt, yt = self._to_device(Xt, yt)

        self.model = engression(
            Xt,
            yt,
            lr=self.learning_rate,
            num_epochs=self.num_epochs,
            num_layer=self.num_layer,
            hidden_dim=self.hidden_dim,
            noise_dim=self.hidden_dim,
            batch_size=self.batch_size,
            resblock=self.resblock,
            device=str(self.device),
        )
        return self

    def predict(self, X):
        Xt = self._prepare(X)
        Xt = Xt.to(self.device)
        out = self.model.predict(Xt, target="mean")
        return out.cpu().numpy().reshape(-1)

    def predict_samples(self, X, sample_size: int = 100):
        Xt = self._prepare(X)
        Xt = Xt.to(self.device)
        samples = []
        for i in range(Xt.size(0)):
            xi = Xt[i : i + 1]
            s = self.model.sample(xi, sample_size=sample_size)
            samples.append(s.cpu().numpy().reshape(-1))
        return np.stack(samples, axis=0)
    
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X)

        residuals = []
        for Xb, yb in train_loader:
            yb_np = yb.cpu().numpy().ravel()
            preds = self.model.predict(Xb.to(self.device), target="mean")
            residuals.append(yb_np - preds.cpu().numpy().ravel())
        sigma = float(np.std(np.concatenate(residuals), ddof=1))

        if sample_size:
            samples = self.predict_samples(X, sample_size)
            return mu, sigma, samples

        return mu, sigma
    



class EngressionClassifier(_TorchBase):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_epochs: int = 500,
        num_layer: int = 3,
        hidden_dim: int = 128,
        resblock: bool = False,
        batch_size: int = 32,
        seed: int = None,
        device: str = None,
    ):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate)
        self.num_epochs = num_epochs
        self.num_layer   = num_layer
        self.hidden_dim  = hidden_dim
        self.resblock    = resblock
        self.seed        = seed
        self.device      = torch.device(device) if device else self.device

    def fit(self, X_train, y_train):
        Xt, yt = self._prepare(X_train, y_train)
        yt = yt.float().view(-1, 1)
        Xt, yt = self._to_device(Xt, yt)

        self.model = engression(
            Xt,
            yt,
            lr=self.learning_rate,
            num_epochs=self.num_epochs,
            num_layer=self.num_layer,
            hidden_dim=self.hidden_dim,
            noise_dim=self.hidden_dim,
            batch_size=self.batch_size,
            resblock=self.resblock,
            device=str(self.device),
        )
        return self

    def predict_proba(self, X):
        Xt = self._prepare(X)
        Xt = Xt.to(self.device)
        raw = self.model.predict(Xt, target="mean")
        probs = torch.sigmoid(raw)
        return probs.cpu().numpy().reshape(-1)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
    

# ----- ResNet Regressor -----
class ResNetRegressor(_TorchBase):
    def __init__(
        self,
        n_blocks: int = 2,
        d_block: int = 128,
        d_hidden_multiplier: float = 1.0,
        dropout1: float = 0.5,
        dropout2: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        patience: int = 10,
        checkpoint_path: str = "checkpoint_resnet.pt",
        seed: int = None,
        n_epochs: int = 100,
    ):
        super().__init__(
            batch_size=batch_size,
            classification=False,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            patience=patience,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.d_hidden_multiplier = d_hidden_multiplier
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # seed everything for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        # prepare loaders
        Xt, yt = self._prepare(X_train, y_train)
        Xt, yt = self._to_device(Xt, yt)
        train_loader = self._loader(Xt, yt, shuffle=True)

        if X_val is not None and y_val is not None:
            Xv, yv = self._prepare(X_val, y_val)
            Xv, yv = self._to_device(Xv, yv)
            val_loader = self._loader(Xv, yv, shuffle=False)
            early_stop = EarlyStopping(self.patience, self.checkpoint_path)
        else:
            val_loader = None
            early_stop = None

        # build model
        self.d_in = Xt.size(1)
        self.model = ResNet(
            d_in=self.d_in,
            d_out=1,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            d_hidden=None,
            d_hidden_multiplier=self.d_hidden_multiplier,
            dropout1=self.dropout1,
            dropout2=self.dropout2,
        ).to(self.device)

        # optimizer + loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        # train
        if val_loader is not None and self.patience:
            epochs_run = train(
                self.model,
                criterion,
                optimizer,
                self.n_epochs,
                train_loader,
                val_loader,
                early_stop,
                self.checkpoint_path,
            )
        else:
            epochs_run = train_no_early_stopping(
                self.model,
                criterion,
                optimizer,
                self.n_epochs,
                train_loader,
            )
        return epochs_run

    def predict(self, X):
        Xt = self._prepare(X).to(self.device)
        loader = self._loader(Xt, torch.zeros_like(Xt[:, :1]), shuffle=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                out = self.model(xb).reshape(-1)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds)
    
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X)  # numpy array, shape (N,)

        self.model.eval()
        residuals = []
        with torch.no_grad():
            for xb, yb in train_loader:
                out = self.model(xb.to(self.device)).view(-1).cpu().numpy()
                residuals.append(yb.cpu().numpy().ravel() - out)
        sigma = float(np.std(np.concatenate(residuals), ddof=1))

        return mu, sigma




class ResNetClassifier(_TorchBase):
    def __init__(
        self,
        num_classes: int,
        n_blocks: int = 2,
        d_block: int = 128,
        d_hidden_multiplier: float = 1.0,
        dropout1: float = 0.5,
        dropout2: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        patience: int = 10,
        checkpoint_path: str = "checkpoint_resnet.pt",
        seed: int = None,
        n_epochs: int = 100,
    ):
        super().__init__(
            batch_size=batch_size,
            classification=True,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            patience=patience,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.d_hidden_multiplier = d_hidden_multiplier
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        
        self.num_classes = num_classes
        self.d_out = self.num_classes
        
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        
        self.model = ResNet(
            d_in=self.d_in,
            d_out=self.d_out,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            d_hidden=None,
            d_hidden_multiplier=self.d_hidden_multiplier,
            dropout1=self.dropout1,
            dropout2=self.dropout2,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()

        if val_loader is not None and self.patience and self.patience > 0:
            early_stop = EarlyStopping(patience=self.patience, path=self.checkpoint_path)
            epochs_run = train(
                self.model,
                criterion,
                optimizer,
                self.n_epochs,
                train_loader,
                val_loader,
                early_stop,
                self.checkpoint_path,
            )
        else:
            epochs_run = train_no_early_stopping(
                self.model,
                criterion,
                optimizer,
                self.n_epochs,
                train_loader,
            )
        return epochs_run

    def predict_proba(self, X):
        Xt = self._prepare(X).to(self.device)
        dummy = torch.zeros((Xt.size(0), 1), dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(Xt, dummy), batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb)
                if self.d_out == 1:
                    probs_batch = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(probs_batch.reshape(-1, self.d_out))
        return np.concatenate(probs, axis=0)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        if self.d_out == 1:
            return (probabilities >= 0.5).astype(int).flatten()
        else:
            return np.argmax(probabilities, axis=1)
    
class FTTrans_Regressor(_TorchBase):
    def __init__(self, cat_cardinalities: list, **kwargs):
        super().__init__(**kwargs)
        self.cat_cardinalities = cat_cardinalities
        self.params = kwargs

    def _prepare_special_data(self, X, y=None):
        # Identify column types based on dataframe dtypes
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        num_cols = X.select_dtypes(exclude=['category', 'object']).columns

        x_num = torch.tensor(X[num_cols].values, dtype=torch.float32)
        x_cat = torch.tensor(X[cat_cols].values, dtype=torch.int64)
        
        if y is None:
            return x_num, x_cat
        
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return x_num, x_cat, y_tensor

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Build Model
        n_num_features = len(X_train.select_dtypes(exclude=['category', 'object']).columns)
        self.model = FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_out=1,
            **self.params
        ).to(self.device)
        
        # Prepare DataLoaders
        x_num_tr, x_cat_tr, y_tr_t = self._to_device(*self._prepare_special_data(X_train, y_train))
        train_ds = TensorDataset(x_num_tr, x_cat_tr, y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            x_num_v, x_cat_v, y_v_t = self._to_device(*self._prepare_special_data(X_val, y_val))
            val_ds = TensorDataset(x_num_v, x_cat_v, y_v_t)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            
        early_stop = EarlyStopping(self.patience, self.checkpoint_path) if val_loader else None
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        return train_trans(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, early_stop, self.checkpoint_path)

    def predict(self, X):
        self.model.eval()
        preds = []
        x_num_te, x_cat_te = self._to_device(*self._prepare_special_data(X))
        test_ds = TensorDataset(x_num_te, x_cat_te)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for x_num, x_cat in test_loader:
                out = self.model(x_num, x_cat).view(-1)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds)


class FTTrans_Classifier(_TorchBase):
    def __init__(self, num_classes: int, cat_cardinalities: list, **kwargs):
        super().__init__(classification=True, **kwargs)
        self.num_classes = num_classes
        self.cat_cardinalities = cat_cardinalities
        self.params = kwargs

    def _prepare_special_data(self, X, y=None):
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        num_cols = X.select_dtypes(exclude=['category', 'object']).columns

        x_num = torch.tensor(X[num_cols].values, dtype=torch.float32)
        x_cat = torch.tensor(X[cat_cols].values, dtype=torch.int64)
        
        if y is None:
            return x_num, x_cat
            
        y_tensor = torch.tensor(y, dtype=torch.long).view(-1)
        return x_num, x_cat, y_tensor

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        n_num_features = len(X_train.select_dtypes(exclude=['category', 'object']).columns)
        self.model = FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_out=self.num_classes,
            **self.params
        ).to(self.device)
        
        x_num_tr, x_cat_tr, y_tr_t = self._to_device(*self._prepare_special_data(X_train, y_train))
        train_ds = TensorDataset(x_num_tr, x_cat_tr, y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            x_num_v, x_cat_v, y_v_t = self._to_device(*self._prepare_special_data(X_val, y_val))
            val_ds = TensorDataset(x_num_v, x_cat_v, y_v_t)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            
        early_stop = EarlyStopping(self.patience, self.checkpoint_path) if val_loader else None
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        return train_trans(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, early_stop, self.checkpoint_path)

    def predict_proba(self, X):
        self.model.eval()
        probas = []
        x_num_te, x_cat_te = self._to_device(*self._prepare_special_data(X))
        test_ds = TensorDataset(x_num_te, x_cat_te)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for x_num, x_cat in test_loader:
                logits = self.model(x_num, x_cat)
                probas.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probas)