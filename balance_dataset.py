import os
import shutil
import random
from PIL import Image
import numpy as np
from collections import defaultdict
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class PokemonDataBalancer:
    def __init__(self, source_dir, target_dir, target_count_per_type=500):

        # Initialize the data balancer
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.target_count = target_count_per_type
        
        # Pokemon type mappings
        self.pokemon_types = {
            'Bulbasaur': ['Grass', 'Poison'],
            'Ivysaur': ['Grass', 'Poison'],
            'Venusaur': ['Grass', 'Poison'],
            'Charmander': ['Fire'],
            'Charmeleon': ['Fire'],
            'Charizard': ['Fire', 'Flying'],
            'Squirtle': ['Water'],
            'Wartortle': ['Water'],
            'Blastoise': ['Water'],
            'Caterpie': ['Bug'],
            'Metapod': ['Bug'],
            'Butterfree': ['Bug', 'Flying'],
            'Weedle': ['Bug', 'Poison'],
            'Kakuna': ['Bug', 'Poison'],
            'Beedrill': ['Bug', 'Poison'],
            'Pidgey': ['Normal', 'Flying'],
            'Pidgeotto': ['Normal', 'Flying'],
            'Pidgeot': ['Normal', 'Flying'],
            'Rattata': ['Normal'],
            'Raticate': ['Normal'],
            'Spearow': ['Normal', 'Flying'],
            'Fearow': ['Normal', 'Flying'],
            'Ekans': ['Poison'],
            'Arbok': ['Poison'],
            'Pikachu': ['Electric'],
            'Raichu': ['Electric'],
            'Sandshrew': ['Ground'],
            'Sandslash': ['Ground'],
            'Nidoran♀': ['Poison'],
            'Nidorina': ['Poison'],
            'Nidoqueen': ['Poison', 'Ground'],
            'Nidoran♂': ['Poison'],
            'Nidorino': ['Poison'],
            'Nidoking': ['Poison', 'Ground'],
            'Clefairy': ['Normal'],
            'Clefable': ['Normal'],
            'Vulpix': ['Fire'],
            'Ninetales': ['Fire'],
            'Jigglypuff': ['Normal'],
            'Wigglytuff': ['Normal'],
            'Zubat': ['Poison', 'Flying'],
            'Golbat': ['Poison', 'Flying'],
            'Oddish': ['Grass', 'Poison'],
            'Gloom': ['Grass', 'Poison'],
            'Vileplume': ['Grass', 'Poison'],
            'Paras': ['Bug', 'Grass'],
            'Parasect': ['Bug', 'Grass'],
            'Venonat': ['Bug', 'Poison'],
            'Venomoth': ['Bug', 'Poison'],
            'Diglett': ['Ground'],
            'Dugtrio': ['Ground'],
            'Meowth': ['Normal'],
            'Persian': ['Normal'],
            'Psyduck': ['Water'],
            'Golduck': ['Water'],
            'Mankey': ['Fighting'],
            'Primeape': ['Fighting'],
            'Growlithe': ['Fire'],
            'Arcanine': ['Fire'],
            'Poliwag': ['Water'],
            'Poliwhirl': ['Water'],
            'Poliwrath': ['Water', 'Fighting'],
            'Abra': ['Psychic'],
            'Kadabra': ['Psychic'],
            'Alakazam': ['Psychic'],
            'Machop': ['Fighting'],
            'Machoke': ['Fighting'],
            'Machamp': ['Fighting'],
            'Bellsprout': ['Grass', 'Poison'],
            'Weepinbell': ['Grass', 'Poison'],
            'Victreebel': ['Grass', 'Poison'],
            'Tentacool': ['Water', 'Poison'],
            'Tentacruel': ['Water', 'Poison'],
            'Geodude': ['Rock', 'Ground'],
            'Graveler': ['Rock', 'Ground'],
            'Golem': ['Rock', 'Ground'],
            'Ponyta': ['Fire'],
            'Rapidash': ['Fire'],
            'Slowpoke': ['Water', 'Psychic'],
            'Slowbro': ['Water', 'Psychic'],
            'Magnemite': ['Electric', 'Steel'],
            'Magneton': ['Electric', 'Steel'],
            "Farfetch'd": ['Normal', 'Flying'],
            'Doduo': ['Normal', 'Flying'],
            'Dodrio': ['Normal', 'Flying'],
            'Seel': ['Water'],
            'Dewgong': ['Water', 'Ice'],
            'Grimer': ['Poison'],
            'Muk': ['Poison'],
            'Shellder': ['Water'],
            'Cloyster': ['Water', 'Ice'],
            'Gastly': ['Ghost', 'Poison'],
            'Haunter': ['Ghost', 'Poison'],
            'Gengar': ['Ghost', 'Poison'],
            'Onix': ['Rock', 'Ground'],
            'Drowzee': ['Psychic'],
            'Hypno': ['Psychic'],
            'Krabby': ['Water'],
            'Kingler': ['Water'],
            'Voltorb': ['Electric'],
            'Electrode': ['Electric'],
            'Exeggcute': ['Grass', 'Psychic'],
            'Exeggutor': ['Grass', 'Psychic'],
            'Cubone': ['Ground'],
            'Marowak': ['Ground'],
            'Hitmonlee': ['Fighting'],
            'Hitmonchan': ['Fighting'],
            'Lickitung': ['Normal'],
            'Koffing': ['Poison'],
            'Weezing': ['Poison'],
            'Rhyhorn': ['Ground', 'Rock'],
            'Rhydon': ['Ground', 'Rock'],
            'Chansey': ['Normal'],
            'Tangela': ['Grass'],
            'Kangaskhan': ['Normal'],
            'Horsea': ['Water'],
            'Seadra': ['Water'],
            'Goldeen': ['Water'],
            'Seaking': ['Water'],
            'Staryu': ['Water'],
            'Starmie': ['Water', 'Psychic'],
            'Mr. Mime': ['Psychic'], 
            'Scyther': ['Bug', 'Flying'],
            'Jynx': ['Ice', 'Psychic'],
            'Electabuzz': ['Electric'],
            'Magmar': ['Fire'],
            'Pinsir': ['Bug'],
            'Tauros': ['Normal'],
            'Magikarp': ['Water'],
            'Gyarados': ['Water', 'Flying'],
            'Lapras': ['Water', 'Ice'],
            'Ditto': ['Normal'],
            'Eevee': ['Normal'],
            'Vaporeon': ['Water'],
            'Jolteon': ['Electric'],
            'Flareon': ['Fire'],
            'Porygon': ['Normal'],
            'Omanyte': ['Rock', 'Water'],
            'Omastar': ['Rock', 'Water'],
            'Kabuto': ['Rock', 'Water'],
            'Kabutops': ['Rock', 'Water'],
            'Aerodactyl': ['Rock', 'Flying'],
            'Snorlax': ['Normal'],
            'Articuno': ['Ice', 'Flying'],
            'Zapdos': ['Electric', 'Flying'],
            'Moltres': ['Fire', 'Flying'],
            'Dratini': ['Dragon'],
            'Dragonair': ['Dragon'],
            'Dragonite': ['Dragon', 'Flying'],
            'Mewtwo': ['Psychic'],
            'Mew': ['Psychic']
        }
        
        self.type_counts = defaultdict(int)
        self.pokemon_by_type = defaultdict(list)
        self.augmentation_transforms = [
            self._rotate_image,
            self._flip_image,
            self._adjust_brightness,
            self._adjust_contrast,
            self._add_noise,
            self._color_jitter,
            self._gaussian_blur
        ]
    
    def _rotate_image(self, img):
        # Rotate image by a random angle
        angle = random.uniform(-30, 30)
        return TF.rotate(img, angle)
    
    def _flip_image(self, img):
        # Randomly flip image horizontally
        return TF.hflip(img)
    
    def _adjust_brightness(self, img):
        # Adjust image brightness
        factor = random.uniform(0.8, 1.2)
        return TF.adjust_brightness(img, factor)
    
    def _adjust_contrast(self, img):
        # Adjust image contrast
        factor = random.uniform(0.8, 1.2)
        return TF.adjust_contrast(img, factor)
    
    def _add_noise(self, img):
        # Add random noise to image
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def _color_jitter(self, img):
        # Apply color jittering
        transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        return transform(img)
    
    def _gaussian_blur(self, img):
        # Apply slight Gaussian blur
        transform = transforms.GaussianBlur(kernel_size=3)
        return transform(img)

    def _create_augmented_image(self, img_path, num_augmentations):
        # Create multiple augmented versions of an image
        original_img = Image.open(img_path).convert('RGB')
        augmented_images = []
        
        for _ in range(num_augmentations):
            # Apply 2-3 random augmentations
            img = original_img.copy()
            num_transforms = random.randint(2, 3)
            selected_transforms = random.sample(self.augmentation_transforms, num_transforms)
            
            for transform in selected_transforms:
                img = transform(img)
            
            augmented_images.append(img)
        
        return augmented_images

    def analyze_dataset(self):
        # Analyze the current distribution of types in the dataset
        print("Analyzing dataset distribution...")
        
        # Reset counts
        self.type_counts.clear()
        self.pokemon_by_type.clear()
        
        # Count images for each type
        for pokemon_name in os.listdir(self.source_dir):
            pokemon_dir = os.path.join(self.source_dir, pokemon_name)
            if not os.path.isdir(pokemon_dir):
                continue
                
            if pokemon_name not in self.pokemon_types:
                print(f"Warning: {pokemon_name} not found in type mappings")
                continue
                
            # Count images for this Pokemon
            image_count = len([f for f in os.listdir(pokemon_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Add to counts for each type
            for type_name in self.pokemon_types[pokemon_name]:
                self.type_counts[type_name] += image_count
                self.pokemon_by_type[type_name].append((pokemon_name, image_count))
        
        print("\nCurrent type distribution:")
        for type_name, count in sorted(self.type_counts.items()):
            print(f"{type_name}: {count} images")
    
    def balance_dataset(self):
        # Create a balanced dataset using augmentation with even distribution among Pokemon
        print("\nBalancing dataset...")
        
        # Create target directory if it doesn't exist
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir)
        
        # First copy all original images to maintain structure
        for pokemon_name in self.pokemon_types:
            source_pokemon_dir = os.path.join(self.source_dir, pokemon_name)
            if not os.path.isdir(source_pokemon_dir):
                continue
                
            target_pokemon_dir = os.path.join(self.target_dir, pokemon_name)
            os.makedirs(target_pokemon_dir)
            
            # Copy original images
            original_images = [f for f in os.listdir(source_pokemon_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in original_images:
                source_path = os.path.join(source_pokemon_dir, img_name)
                target_path = os.path.join(target_pokemon_dir, img_name)
                shutil.copy2(source_path, target_path)
        
        # Process each type
        for type_name, current_count in self.type_counts.items():
            print(f"\nProcessing {type_name} type...")
            needed_images = self.target_count - current_count
            
            if needed_images <= 0:
                print(f"Already have enough {type_name} type images")
                continue
                
            print(f"Need to generate {needed_images} additional images")
            
            # Get all Pokemon of this type
            pokemon_of_type = self.pokemon_by_type[type_name]
            
            # Calculate total available source images for this type
            total_source_images = sum(count for _, count in pokemon_of_type)
            
            # Calculate augmentations needed per original image to reach target
            augmentations_per_image = needed_images / total_source_images
            
            print(f"Will create approximately {augmentations_per_image:.1f} augmented versions per original image")
            
            # Generate augmented images for each Pokemon of this type
            for pokemon_name, pokemon_image_count in pokemon_of_type:
                # Calculate exact number of augmentations needed for this Pokemon
                pokemon_augmentations = int(np.ceil(augmentations_per_image * pokemon_image_count))
                augmentations_per_source = int(np.ceil(pokemon_augmentations / pokemon_image_count))
                
                source_pokemon_dir = os.path.join(self.source_dir, pokemon_name)
                target_pokemon_dir = os.path.join(self.target_dir, pokemon_name)
                
                # Get list of original images
                original_images = [f for f in os.listdir(source_pokemon_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"\nProcessing {pokemon_name}: creating {pokemon_augmentations} new images "
                    f"({augmentations_per_source} versions per source image)")
                
                # Create augmented versions
                augmentations_created = 0
                for img_name in original_images:
                    if augmentations_created >= pokemon_augmentations:
                        break
                        
                    source_path = os.path.join(source_pokemon_dir, img_name)
                    augmented_images = self._create_augmented_image(
                        source_path, 
                        min(augmentations_per_source, pokemon_augmentations - augmentations_created)
                    )
                    
                    # Save augmented images
                    for i, aug_img in enumerate(augmented_images):
                        aug_name = f"aug_{augmentations_created + i}_{img_name}"
                        aug_path = os.path.join(target_pokemon_dir, aug_name)
                        aug_img.save(aug_path)
                    
                    augmentations_created += len(augmented_images)
        
        print("\nDataset balancing complete!")

def main():
    # Set paths
    source_dir = "./dataset" # OG dataset
    target_dir = "./balanced-dataset"  # NEW dataset
    
    # Create balancer
    balancer = PokemonDataBalancer(source_dir, target_dir, target_count_per_type=500)
    
    # Analyze current distribution
    balancer.analyze_dataset()
    
    # Balance dataset
    balancer.balance_dataset()
    
    # Verify final distribution
    print("\nVerifying final distribution...")
    balancer = PokemonDataBalancer(target_dir, "")
    balancer.analyze_dataset()

if __name__ == "__main__":
    main()