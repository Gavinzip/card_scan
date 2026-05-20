# Reference Images on a Private Server

The FAISS indexes store provenance paths such as:

```text
/Users/xiecongfeng/card_data/raw/pokemon/Pokemon TCG/Pokemon TCG/151/sv3-5_en_001_std.jpg
```

Those paths are not public URLs. To display English Kaggle images in the web UI, copy the image folder to the server and expose it through the API as read-only static files.

## Recommended Layout

On the private server:

```bash
sudo mkdir -p /data/card_scan/reference_images/pokemon_en
```

Copy the image files from this machine. The source should be the inner folder that contains set folders such as `151/`, `brilliant-stars/`, and `pop-series-4/`:

```bash
rsync -a --info=progress2 \
  "/Users/xiecongfeng/card_data/raw/pokemon/Pokemon TCG/Pokemon TCG/" \
  user@YOUR_SERVER:/data/card_scan/reference_images/pokemon_en/
```

If you prefer a tarball:

```bash
tar -C "/Users/xiecongfeng/card_data/raw/pokemon/Pokemon TCG/Pokemon TCG" \
  -cf pokemon_en_images.tar .

scp pokemon_en_images.tar user@YOUR_SERVER:/data/card_scan/

ssh user@YOUR_SERVER
mkdir -p /data/card_scan/reference_images/pokemon_en
tar -C /data/card_scan/reference_images/pokemon_en -xf /data/card_scan/pokemon_en_images.tar
```

## Zeabur Volume from Google Drive

Mount a Zeabur Volume at `/data`, then use Zeabur file management or a shell to place the tarball in the volume.

If using a Google Drive share link:

```bash
python -m pip install gdown
mkdir -p /data/reference_images/pokemon_en
gdown --fuzzy "GOOGLE_DRIVE_SHARE_URL" -O /data/pokemon_en_images.tar
tar -xf /data/pokemon_en_images.tar -C /data/reference_images/pokemon_en
rm /data/pokemon_en_images.tar
find /data/reference_images/pokemon_en -maxdepth 2 -type f | head
```

Set these environment variables on the Zeabur service:

```text
CARD_SCAN_IMAGE_ROOTS=pokemon_en=/data/reference_images/pokemon_en
CARD_SCAN_LOCAL_PATH_REWRITES=/Users/xiecongfeng/card_data/raw/pokemon/Pokemon TCG/Pokemon TCG=/data/reference_images/pokemon_en
```

After redeploy, `/health` should show `reference_images.pokemon_en.exists` as `true`.

## Docker Run

Mount the copied image folder into the container and tell the API how to rewrite the old local paths:

```bash
docker run --rm -p 8080:8080 \
  -v /data/card_scan/reference_images/pokemon_en:/app/reference_images/pokemon_en:ro \
  -e CARD_SCAN_IMAGE_ROOTS="pokemon_en=/app/reference_images/pokemon_en" \
  -e CARD_SCAN_LOCAL_PATH_REWRITES="/Users/xiecongfeng/card_data/raw/pokemon/Pokemon TCG/Pokemon TCG=/app/reference_images/pokemon_en" \
  card_scan:latest
```

Then check:

```bash
curl http://localhost:8080/health
```

The response should include:

```json
{
  "reference_images": {
    "pokemon_en": {
      "exists": true,
      "route": "/reference-images/pokemon_en"
    }
  }
}
```

When a search result comes from a local Kaggle image, `/recognize` will return:

```json
{
  "image_url": null,
  "reference_image_url": "/reference-images/pokemon_en/pop-series-4/mudkip-pop-series-4-11.jpg",
  "display_image_url": "/reference-images/pokemon_en/pop-series-4/mudkip-pop-series-4-11.jpg"
}
```

The frontend uses `display_image_url`, so English images will render.

## Docker Compose Example

```yaml
services:
  card-scan:
    image: card_scan:latest
    ports:
      - "8080:8080"
    volumes:
      - /data/card_scan/reference_images/pokemon_en:/app/reference_images/pokemon_en:ro
    environment:
      CARD_SCAN_IMAGE_ROOTS: pokemon_en=/app/reference_images/pokemon_en
      CARD_SCAN_LOCAL_PATH_REWRITES: /Users/xiecongfeng/card_data/raw/pokemon/Pokemon TCG/Pokemon TCG=/app/reference_images/pokemon_en
      CARD_SCAN_PRELOAD: "false"
      CARD_SCAN_DEVICE: cpu
```

## Why Not Commit Images?

Do not put the 3.59 GB image folder into GitHub or the normal Zeabur build context. It makes pushes and deployments slow, risks hitting hosting limits, and forces every deploy to rebuild with the full image payload. A mounted folder or object storage bucket is easier to update and cheaper to operate.
